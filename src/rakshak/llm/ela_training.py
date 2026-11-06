"""Training-only synthetic ELA packages and verifier-gated open-model paraphrases.

No SMS corpus, test/external rows, or assistant-written target sentences are accepted.
Targets come from the existing deterministic template bank or the pinned Llama teacher.
Generation and fine-tuning belong in Kaggle jobs; this module imports no model runtime.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass
from functools import lru_cache
from importlib import resources
from pathlib import Path

from rakshak import prompts, verifier
from rakshak.canonicalize import canonicalize
from rakshak.ela_eval import score_advisory
from rakshak.evidence import EvidencePackage, build_package
from rakshak.llama_source import UNSLOTH_REPO, UNSLOTH_REVISION
from rakshak.rules import EVIDENCE_TYPES, Evidence


def _digest(value) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    return hashlib.sha256(payload).hexdigest()


@lru_cache(maxsize=1)
def template_fingerprint() -> str:
    """Identify the existing templates, lexical gate and prompt that produced the targets."""
    files = [Path(verifier.__file__), Path(prompts.__file__), Path(__file__)]
    contents = [p.read_text(encoding="utf-8") for p in files]
    contents.append(resources.files("rakshak").joinpath("data/ela_lexicon.json").read_text())
    return _digest(contents)


def composition_key(package: EvidencePackage) -> str:
    """All languages, orderings and risk variants of a type combination share one group."""
    return _digest(sorted(e.type for e in package.evidence))


def development_partition(package: EvidencePackage, seed: int = 13) -> str:
    """Synthetic development only; singletons/empty packages train for vocabulary coverage."""
    if len(package.evidence) <= 1:
        return "train"
    digest = _digest([seed, composition_key(package)])
    return "dev" if int(digest[:16], 16) % 5 == 0 else "train"


def synthetic_packages(n: int = 2000, seed: int = 13) -> list[EvidencePackage]:
    """Abstract evidence combinations, never generated or relabelled SMS test messages.

    Vary the abstract signal strengths and confidence state to exercise the existing risk
    ladder. These are synthetic verbalisation inputs, not examples for detector training.
    """
    if n < len(EVIDENCE_TYPES) + 1:
        raise ValueError("n must cover the empty package and every evidence singleton")
    rng = random.Random(seed)
    combinations = [[], *[[t] for t in EVIDENCE_TYPES]]
    while len(combinations) < n:
        width = rng.randint(2, 6)
        combinations.append(
            rng.choices(EVIDENCE_TYPES, k=width) if rng.random() < 0.25
            else rng.sample(EVIDENCE_TYPES, width)
        )
    packages = []
    canon = canonicalize("")
    for types in combinations:
        evidence = [
            Evidence(f"E{i + 1}", t, (0, 0), "",
                     0.4 if t == "urgency" else rng.choice((0.55, 0.7, 0.9)))
            for i, t in enumerate(types)
        ]
        state = rng.choice(("benign", "uncertain", "scam")) if types else "benign"
        packages.append(build_package(canon, evidence, state))
    return packages


@dataclass(frozen=True)
class TrainingExample:
    example_id: str
    composition: str
    partition: str
    lang: str
    prompt: str
    target: str
    target_source: str
    template_sha256: str
    evidence_types: tuple[str, ...]
    teacher_repo: str | None = None
    teacher_revision: str | None = None
    teacher_seed: int | None = None
    parent_ids: tuple[str, ...] = ()
    synthetic: bool = True
    training_only: bool = True

    def to_dict(self):
        return asdict(self)


def _example(package, lang, points, source, seed, teacher=None, parent_ids=()):
    if (package.urls or package.domains or package.phones or package.brand_domains
            or any(e.text or e.span != (0, 0) for e in package.evidence)):
        raise ValueError("training inputs must be abstract synthetic packages without message text")
    target = json.dumps({
        "lang": lang, "summary": verifier._template_summary(package, lang), "points": points,
    }, ensure_ascii=False)
    if any(token in target for token in (prompts.BOS, prompts.EOT, prompts.SOT, prompts.EOT_HDR)):
        raise ValueError("training target rejected: chat control token")
    verdict = verifier.check_points(target, package, lang)
    score = score_advisory(target, package, lang)
    if not verdict.ok or score["unsupported_n"] or score["safe_claim"]:
        problems = [*verdict.schema_problems, *verdict.summary_problems,
                    *(p for point in verdict.points for p in point.problems),
                    *score["unsupported"]]
        if score["safe_claim"]:
            problems.append("training target must not reassure that a message is safe")
        raise ValueError("training target rejected: " + "; ".join(problems))
    prompt = prompts.build_points_chat_prompt(package, lang)
    metadata = teacher or {}
    return TrainingExample(
        example_id=_digest([prompt, target]),
        composition=composition_key(package), partition=development_partition(package, seed),
        lang=lang, prompt=prompt, target=target, target_source=source,
        template_sha256=template_fingerprint(),
        evidence_types=tuple(e.type for e in package.evidence), parent_ids=tuple(parent_ids),
        teacher_repo=metadata.get("repo"), teacher_revision=metadata.get("revision"),
        teacher_seed=metadata.get("seed"),
    )


def template_example(package: EvidencePackage, lang: str, *, variant=0, seed=13):
    """Select only strings already in the repository's deterministic candidate bank."""
    points = [verifier.rank_candidates(e.type, lang)[variant % 4] for e in package.evidence]
    return _example(package, lang, points, "deterministic_template", seed)


def llama_paraphrase_example(
    package: EvidencePackage, lang: str, output: str, *,
    teacher_repo: str, teacher_revision: str, teacher_seed: int,
    generated_on: str, seed: int = 13,
):
    """Accept points from the pinned teacher's Kaggle generation, then apply the same gate.

    Summary targets stay deterministic. Unknown provenance, extra fields, unsupported
    actions/claims and lexical mismatches are rejected, never silently repaired into data.
    """
    if (teacher_repo, teacher_revision, generated_on) != (UNSLOTH_REPO, UNSLOTH_REVISION, "kaggle"):
        raise ValueError("only the pinned Llama teacher's Kaggle generation is permitted")
    obj = json.loads(output)
    if not isinstance(obj, dict) or set(obj) != {"points"}:
        raise ValueError("teacher must return exactly the points field")
    return _example(package, lang, obj["points"], "llama_paraphrase", seed, {
        "repo": teacher_repo, "revision": teacher_revision, "seed": teacher_seed,
    })


def assert_composition_disjoint(examples):
    groups = {}
    for example in examples:
        if (not example.synthetic or not example.training_only
                or example.partition not in {"train", "dev"}):
            raise ValueError("only synthetic training/development examples are permitted")
        if example.target_source not in {
            "deterministic_template", "llama_paraphrase", "verified_composition",
        }:
            raise ValueError("unknown training target provenance")
        if example.teacher_repo is not None and (
            example.teacher_repo, example.teacher_revision
        ) != (UNSLOTH_REPO, UNSLOTH_REVISION):
            raise ValueError("unapproved teacher provenance")
        previous = groups.setdefault(example.composition, example.partition)
        if previous != example.partition:
            raise ValueError("one evidence composition appears in train and dev")


def compose_example(package: EvidencePackage, lang: str, parents, *, seed=13):
    """Compose only already gated singleton targets; keep every parent for auditability."""
    if len(parents) != len(package.evidence):
        raise ValueError("one gated singleton parent per evidence point is required")
    assert_composition_disjoint(parents)
    points = []
    teacher = None
    for evidence, parent in zip(package.evidence, parents, strict=True):
        if parent.lang != lang or parent.evidence_types != (evidence.type,):
            raise ValueError("parent language/evidence type does not match the synthetic point")
        point = json.loads(parent.target)["points"]
        if len(point) != 1:
            raise ValueError("composition parents must be singleton targets")
        points.append(point[0])
        if parent.teacher_repo:
            teacher = {"repo": parent.teacher_repo, "revision": parent.teacher_revision}
    return _example(package, lang, points, "verified_composition", seed, teacher,
                    parent_ids=[p.example_id for p in parents])


def deduplicate_examples(examples):
    """Copies across teacher/template paths do not receive extra training weight."""
    assert_composition_disjoint(examples)
    unique = {}
    for example in examples:
        unique.setdefault(example.example_id, example)
    return list(unique.values())


def completion_features(example, tokenizer, *, max_length=768):
    """Mask all prompt tokens; reject oversized targets instead of silently truncating JSON."""
    assert_composition_disjoint([example])
    prefix = tokenizer(example.prompt, add_special_tokens=False)["input_ids"]
    completion = tokenizer(example.target + prompts.EOT, add_special_tokens=False)["input_ids"]
    if len(prefix) + len(completion) > max_length:
        raise ValueError("training example exceeds max_length; truncation is not permitted")
    if not prefix or not completion:
        raise ValueError("prompt and target must both tokenize to nonempty sequences")
    return {
        "input_ids": prefix + completion,
        "attention_mask": [1] * (len(prefix) + len(completion)),
        "labels": [-100] * len(prefix) + completion,
    }
