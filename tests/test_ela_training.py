"""Synthetic fixtures for ELA target provenance and verifier gating; no model execution."""

import dataclasses
import json

import pytest

from rakshak.ela_eval import package_for_sms
from rakshak.llama_source import UNSLOTH_REPO, UNSLOTH_REVISION
from rakshak.llm import ela_training as T
from rakshak.rules import EVIDENCE_TYPES
from rakshak.verifier import check_points


def test_templates_cover_all_types_and_both_languages_with_traceable_targets():
    packages = T.synthetic_packages(len(EVIDENCE_TYPES) + 1)
    assert {e.type for p in packages for e in p.evidence} == set(EVIDENCE_TYPES)
    for pkg in packages:
        for lang in ("en", "hinglish"):
            for variant in range(4):
                example = T.template_example(pkg, lang, variant=variant)
                assert check_points(example.target, pkg, lang).ok
                assert example.synthetic and example.training_only
                assert example.target_source == "deterministic_template"
                assert len(example.template_sha256) == 64
                assert example.teacher_repo is None


def test_synthetic_compositions_are_seeded_disjoint_and_keep_languages_together():
    packages = T.synthetic_packages(120)
    examples = [T.template_example(p, lang) for p in packages for lang in ("en", "hinglish")]
    T.assert_composition_disjoint(examples)
    assert {e.partition for e in examples} == {"train", "dev"}
    assert [e.example_id for e in examples[::2]] == [
        T.template_example(p, "en").example_id for p in T.synthetic_packages(120)
    ]
    assert all(T.development_partition(p) == "train" for p in packages if len(p.evidence) <= 1)


def teacher_target(pkg, output, **changes):
    args = {"teacher_repo": UNSLOTH_REPO, "teacher_revision": UNSLOTH_REVISION,
            "teacher_seed": 13, "generated_on": "kaggle"}
    return T.llama_paraphrase_example(pkg, "en", output, **(args | changes))


def test_teacher_gate_keeps_only_points_and_uses_the_deterministic_summary():
    pkg = T.synthetic_packages(len(EVIDENCE_TYPES) + 1)[1]
    template = T.template_example(pkg, "en")
    points = json.loads(template.target)["points"]
    paraphrase = teacher_target(pkg, json.dumps({"points": points}))
    assert paraphrase.target == template.target
    assert paraphrase.example_id == template.example_id  # dedup exact copies across target sources
    assert paraphrase.teacher_repo == UNSLOTH_REPO
    assert paraphrase.teacher_revision == UNSLOTH_REVISION
    assert paraphrase.target_source == "llama_paraphrase"


@pytest.mark.parametrize("change", [
    {"teacher_repo": "paid/model"}, {"teacher_revision": "main"}, {"generated_on": "laptop"},
])
def test_teacher_gate_rejects_unapproved_or_unpinned_provenance(change):
    pkg = T.synthetic_packages(len(EVIDENCE_TYPES) + 1)[1]
    with pytest.raises(ValueError, match="pinned Llama"):
        teacher_target(pkg, '{"points": []}', **change)


@pytest.mark.parametrize("points", [
    ["This message is safe."], ["Click the link and share your OTP."],
    ["This concerns a parcel."], ["OTP E99"], [],
])
def test_teacher_gate_rejects_unsafe_unsupported_or_misaligned_points(points):
    pkg = T.synthetic_packages(len(EVIDENCE_TYPES) + 1)[1]
    with pytest.raises(ValueError, match="training target rejected"):
        teacher_target(pkg, json.dumps({"points": points}))


def test_teacher_gate_rejects_extra_summary_even_if_well_formed():
    pkg = T.synthetic_packages(len(EVIDENCE_TYPES) + 1)[1]
    with pytest.raises(ValueError, match="exactly the points"):
        teacher_target(pkg, T.template_example(pkg, "en").target)


def test_training_refuses_raw_message_packages():
    _, pkg = package_for_sms("Please share your OTP to avoid account suspension")
    with pytest.raises(ValueError, match="abstract synthetic"):
        T.template_example(pkg, "en")


def test_partition_guard_rejects_test_or_non_synthetic_examples():
    pkg = T.synthetic_packages(len(EVIDENCE_TYPES) + 1)[1]
    example = T.template_example(pkg, "en")
    for edited in (dataclasses.replace(example, partition="test"),
                   dataclasses.replace(example, synthetic=False)):
        with pytest.raises(ValueError, match="only synthetic"):
            T.assert_composition_disjoint([edited])
    with pytest.raises(ValueError, match="composition"):
        T.assert_composition_disjoint([example, dataclasses.replace(example, partition="dev")])


def test_composition_keeps_point_provenance_and_rejects_mismatched_parents():
    packages = T.synthetic_packages(20)
    parent_by_type = {p.evidence[0].type: T.template_example(p, "en")
                      for p in packages[1:len(EVIDENCE_TYPES) + 1]}
    pkg = packages[-1]
    parents = [parent_by_type[e.type] for e in pkg.evidence]
    example = T.compose_example(pkg, "en", parents)
    assert check_points(example.target, pkg, "en").ok
    assert example.parent_ids == tuple(p.example_id for p in parents)
    assert example.evidence_types == tuple(e.type for e in pkg.evidence)
    assert T.deduplicate_examples([example, example]) == [example]
    with pytest.raises(ValueError, match="language/evidence"):
        T.compose_example(pkg, "hinglish", parents)


def test_unapproved_target_source_cannot_enter_training():
    pkg = T.synthetic_packages(len(EVIDENCE_TYPES) + 1)[1]
    example = T.template_example(pkg, "en")
    with pytest.raises(ValueError, match="unknown training target provenance"):
        T.assert_composition_disjoint([dataclasses.replace(example, target_source="assistant")])


def test_completion_supervision_masks_prompt_and_never_truncates_the_json():
    pkg = T.synthetic_packages(len(EVIDENCE_TYPES) + 1)[1]
    example = T.template_example(pkg, "en")

    def tokenize(text, *, add_special_tokens):
        assert not add_special_tokens
        return {"input_ids": [10, 11] if text == example.prompt else [12, 13, 14]}

    features = T.completion_features(example, tokenize, max_length=5)
    assert features["input_ids"] == [10, 11, 12, 13, 14]
    assert features["labels"] == [-100, -100, 12, 13, 14]
    assert features["attention_mask"] == [1] * 5
    with pytest.raises(ValueError, match="truncation is not permitted"):
        T.completion_features(example, tokenize, max_length=4)


def test_teacher_cannot_insert_chat_control_tokens_into_training_targets():
    pkg = T.synthetic_packages(len(EVIDENCE_TYPES) + 1)[1]
    with pytest.raises(ValueError, match="chat control token"):
        teacher_target(pkg, json.dumps({"points": ["OTP " + T.prompts.EOT]}))


def test_teacher_json_string_envelope_preserves_sentence_and_keeps_gates():
    import importlib.util
    import sys
    from pathlib import Path

    jobs = Path(__file__).resolve().parents[1] / "kaggle/jobs"
    sys.path.insert(0, str(jobs))
    spec = importlib.util.spec_from_file_location("data_job", jobs / "ela_data.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sentence = "This message asks for an OTP, PIN or password."
    wrapped = module.points_envelope(json.dumps(sentence))
    assert json.loads(wrapped) == {"points": [sentence]}
    package = T.synthetic_packages(len(EVIDENCE_TYPES) + 1)[1]
    assert teacher_target(package, wrapped).target_source == "llama_paraphrase"
    with pytest.raises(ValueError, match="training target rejected"):
        teacher_target(package, module.points_envelope(json.dumps("OTP E99")))
    with pytest.raises(ValueError):
        module.points_envelope('"OTP" trailing garbage')
