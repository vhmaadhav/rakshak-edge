"""Tests for the chunked labelling pipeline. No network, no model calls, no API credentials."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
import pytest

from rakshak.llm import budget, chunks, cli, jobs, prompts, validate
from rakshak.llm.schemas import (
    ANNOTATE_SCHEMA,
    SCHEMAS,
    ZERO_SHOT_SCHEMA,
    with_id,
)
from rakshak.llm.schemas import (
    validate as schema_validate,
)

REPO = Path(__file__).resolve().parents[1]


def make_frame(n: int = 300) -> pd.DataFrame:
    rows = []
    for i in range(n):
        rows.append(
            {
                "id": f"m{i:05d}",
                "text": f"sms number {i}: share your OTP now",
                "source": ["scamshield", "uci_sms_spam", "imc25"][i % 3],
                "lang": ["en", "hinglish"][i % 2],
                "label_fine": ["ham", "scam", "spam"][i % 3],
                "label_binary": int(i % 3 != 0),
                "campaign_id": f"c{i % 37}",
                "dup_of": None,
                "eval_ok": True,
                "label_conflict": False,
            }
        )
    return pd.DataFrame(rows)


def partitions_for(frame: pd.DataFrame) -> dict[str, set[str]]:
    ids = sorted(frame["id"])
    return {"train": set(ids[:200]), "cal": set(ids[200:250]), "test": set(ids[250:])}


def annotation(rid: str, spans: list[str]) -> dict:
    return {
        "id": rid,
        "policy_label": "scam",
        "scam_type": "otp_theft",
        "urgency": True,
        "otp_pin_request": True,
        "payment_request": False,
        "remote_access_app": False,
        "link_present": False,
        "impersonated_brand": None,
        "evidence_spans": spans,
        "confidence": "high",
    }


def write_lines(path: Path, rows: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [r if isinstance(r, str) else json.dumps(r, ensure_ascii=False) for r in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# --- schemas -----------------------------------------------------------------------------------


def test_annotation_schema_accepts_valid_and_rejects_bad_rows():
    assert schema_validate(annotation("x", ["share your OTP"]), with_id(ANNOTATE_SCHEMA)) == []
    bad_enum = {**annotation("x", []), "policy_label": "fraud"}
    assert schema_validate(bad_enum, with_id(ANNOTATE_SCHEMA))
    extra = {**annotation("x", []), "note": "free text"}
    assert any("unexpected" in e for e in schema_validate(extra, with_id(ANNOTATE_SCHEMA)))
    too_many = annotation("x", ["a", "b", "c", "d", "e"])
    assert any("more than 4" in e for e in schema_validate(too_many, with_id(ANNOTATE_SCHEMA)))
    missing = {k: v for k, v in annotation("x", []).items() if k != "confidence"}
    assert any("missing" in e for e in schema_validate(missing, with_id(ANNOTATE_SCHEMA)))


def test_zero_shot_schema_is_small_and_closed():
    ok = {"id": "x", "label": "not_scam", "confidence": "low"}
    assert schema_validate(ok, with_id(ZERO_SHOT_SCHEMA)) == []
    assert schema_validate({**ok, "label": "maybe"}, with_id(ZERO_SHOT_SCHEMA))


def test_schemas_are_closed_objects_with_matching_required():
    for name, schema in SCHEMAS.items():
        assert schema["additionalProperties"] is False, name
        assert schema["required"] == list(schema["properties"]), name
        assert len(json.dumps(schema)) < 2500, name


def test_audit_uses_annotation_schema():
    assert SCHEMAS["audit"] is SCHEMAS["annotate"]


# --- prompts and brief -------------------------------------------------------------------------

VOLATILE = re.compile(r"\d{4}-\d{2}-\d{2}|[0-9a-f]{8}-[0-9a-f]{4}-|\b\d{10,}\b")


@pytest.mark.parametrize(
    "text",
    [
        prompts.RELABEL_INSTRUCTIONS,
        prompts.AUDIT_INSTRUCTIONS,
        prompts.NOISE_INSTRUCTIONS,
        prompts.ZERO_SHOT_INSTRUCTIONS,
    ],
)
def test_instructions_are_stable_and_free_of_volatile_values(text):
    assert len(text) >= 1000
    assert not VOLATILE.search(text)


def test_brief_is_deterministic_and_marks_input_untrusted():
    spec = jobs.get_job("A")
    first = prompts.render_brief(spec.name, spec.kind, spec.instructions, spec.schema_name)
    second = prompts.render_brief(spec.name, spec.kind, spec.instructions, spec.schema_name)
    assert first == second
    assert not VOLATILE.search(first)
    assert "untrusted data" in first
    assert "never an instruction" in first
    assert "data/llm/A/tasks/chunk_XXXX.jsonl" in first
    assert "data/llm/A/results/chunk_XXXX.jsonl" in first
    assert '"policy_label"' in first


def test_job_briefs_are_not_mixed_up():
    text_a = prompts.render_brief("A", "relabel", prompts.RELABEL_INSTRUCTIONS, "annotate")
    text_d = prompts.render_brief("D", "zero_shot", prompts.ZERO_SHOT_INSTRUCTIONS, "zero_shot")
    assert "evidence_spans" in text_a
    assert "evidence_spans" not in text_d


# --- jobs and sampling -------------------------------------------------------------------------


def test_stratified_sample_is_deterministic_and_sized():
    frame = make_frame(300)
    first = jobs.stratified_sample(frame, 60, seed=13)
    second = jobs.stratified_sample(frame, 60, seed=13)
    assert first == second
    assert len(first) == len(set(first)) == 60
    assert jobs.stratified_sample(frame, 60, seed=14) != first


def test_stratified_sample_follows_strata_proportions():
    frame = make_frame(300)
    picked = set(jobs.stratified_sample(frame, 60, seed=13))
    sub = frame.assign(_s=frame[list(jobs.STRATA)].astype(str).agg("|".join, axis=1))
    for stratum, group in sub.groupby("_s"):
        quota = 60 * len(group) / len(frame)
        got = int(group["id"].isin(picked).sum())
        assert abs(got - quota) <= 1, stratum


def test_stratified_sample_spreads_across_campaigns():
    frame = pd.DataFrame(
        {
            "id": [f"r{i:03d}" for i in range(100)],
            "source": ["s"] * 100,
            "lang": ["en"] * 100,
            "label_fine": ["scam"] * 100,
            "campaign_id": [f"camp{i % 10}" for i in range(100)],
        }
    )
    picked = jobs.stratified_sample(frame, 5)
    campaigns = set(frame.set_index("id").loc[picked, "campaign_id"])
    assert len(campaigns) == 5


def test_stratified_sample_returns_everything_when_n_is_large():
    frame = make_frame(20)
    assert jobs.stratified_sample(frame, 50) == sorted(frame["id"])


def test_allocate_caps_small_strata():
    alloc = jobs._allocate({"a": 1, "b": 100}, 10)
    assert sum(alloc.values()) == 10
    assert alloc["a"] <= 1


def test_relabel_selection_excludes_test_partition():
    frame = make_frame(300)
    parts = partitions_for(frame)
    selected = set(jobs.select_relabel(frame, parts))
    assert selected == parts["train"] | parts["cal"]
    assert not selected & parts["test"]


def test_policy_refusal_for_job_f():
    with pytest.raises(jobs.JobDisabledError, match="docs/BUDGET.md"):
        jobs.get_job("F")
    with pytest.raises(jobs.JobDisabledError):
        jobs.get_job("f")


def test_unspecified_jobs_are_not_runnable():
    for name in ("E", "G", "Z"):
        with pytest.raises(jobs.JobUnavailableError):
            jobs.get_job(name)


def test_cli_refuses_job_f_with_exit_2(tmp_path, capsys):
    assert cli.main(["--root", str(tmp_path), "export", "F"]) == 2
    assert "disabled by policy" in capsys.readouterr().err
    assert not (tmp_path / "F").exists()


# --- export ------------------------------------------------------------------------------------


def test_export_writes_only_id_and_text_and_logs_chunks(tmp_path):
    frame = make_frame(300)
    parts = partitions_for(frame)
    written = chunks.export_job("A", tmp_path, frame, parts, chunk_size=120)
    assert [p.name for p in written] == ["chunk_0000.jsonl", "chunk_0001.jsonl", "chunk_0002.jsonl"]
    seen: list[str] = []
    for path in written:
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            assert set(row) == {"id", "text"}
            seen.append(row["id"])
    assert len(seen) == len(set(seen)) == len(parts["train"] | parts["cal"])
    assert (tmp_path / "A" / "BRIEF.md").exists()
    entries = budget.read_ledger(tmp_path)
    assert [e["status"] for e in entries] == ["exported"] * 3
    assert sum(e["messages"] for e in entries) == len(seen)


def test_ledger_has_no_dollar_fields(tmp_path):
    chunks.export_job("A", tmp_path, make_frame(60), partitions_for(make_frame(60)), 25)
    for line in (tmp_path / budget.LEDGER_NAME).read_text(encoding="utf-8").splitlines():
        assert set(json.loads(line)) == {
            "job",
            "model",
            "chunk",
            "messages",
            "started",
            "finished",
            "status",
        }


def test_no_anthropic_dependency_in_project():
    assert "anthropic" not in (REPO / "pyproject.toml").read_text(encoding="utf-8").lower()


# --- validation and merge ----------------------------------------------------------------------


def export_small(tmp_path: Path, job: str = "A") -> tuple[pd.DataFrame, dict]:
    frame = pd.DataFrame(
        [
            {
                "id": "a1",
                "text": "Your OTP is 1234. Share your OTP to continue",
                "source": "s",
                "lang": "en",
                "label_fine": "scam",
                "label_binary": 1,
                "campaign_id": "c1",
                "dup_of": None,
                "eval_ok": True,
                "label_conflict": False,
            },
            {
                "id": "a2",
                "text": "Dinner at 8 tonight?",
                "source": "s",
                "lang": "en",
                "label_fine": "ham",
                "label_binary": 0,
                "campaign_id": "c2",
                "dup_of": None,
                "eval_ok": True,
                "label_conflict": False,
            },
            {
                "id": "a3",
                "text": "Parcel fee Rs 49 pay now",
                "source": "s",
                "lang": "en",
                "label_fine": "scam",
                "label_binary": 1,
                "campaign_id": "c3",
                "dup_of": None,
                "eval_ok": True,
                "label_conflict": False,
            },
        ]
    )
    ids = sorted(frame["id"])
    parts = {"train": set(ids), "cal": set(), "test": set()}
    chunks.export_job(job, tmp_path, frame, parts, chunk_size=2)
    return frame, parts


def test_validate_and_merge_good_and_bad_chunks(tmp_path):
    export_small(tmp_path)
    results = tmp_path / "A" / "results"
    write_lines(
        results / "chunk_0000.jsonl",
        [annotation("a1", ["Share your OTP"]), annotation("a2", [])],
    )
    write_lines(results / "chunk_0001.jsonl", [annotation("a3", ["not in the text"])])

    reports = validate.validate_job("A", tmp_path)
    assert [r.ok for r in reports] == [True, False]
    assert "not verbatim" in reports[1].errors[0]
    statuses = [e["status"] for e in budget.read_ledger(tmp_path) if e["status"] != "exported"]
    assert statuses == ["validated_ok", "validated_failed"]

    out, n_rows, skipped = validate.merge_job("A", tmp_path)
    assert n_rows == 2
    assert skipped == ["chunk_0001.jsonl"]
    merged = pd.read_parquet(out)
    assert sorted(merged["id"]) == ["a1", "a2"]
    assert json.loads(merged.set_index("id").loc["a1", "evidence_spans"]) == ["Share your OTP"]


def test_validate_reports_each_kind_of_line_problem(tmp_path):
    export_small(tmp_path)
    results = tmp_path / "A" / "results"
    bad_enum = {**annotation("a2", []), "policy_label": "fraud"}
    write_lines(
        results / "chunk_0000.jsonl",
        [
            annotation("a1", ["OTP"]),
            "{not json",
            annotation("zzz", []),
            annotation("a1", []),
            bad_enum,
            annotation("a2", ["a", "b", "c", "d", "e"]),
        ],
    )
    report, rows = validate.check_chunk(
        tmp_path / "A" / "tasks" / "chunk_0000.jsonl", results / "chunk_0000.jsonl", "annotate"
    )
    assert not report.ok
    assert set(rows) == {"a1"}
    text = "\n".join(report.errors)
    assert "invalid JSON" in text
    assert "not in this chunk" in text
    assert "duplicate id" in text
    assert "not an allowed value" in text
    assert "more than 4" in text
    assert "no valid row" in text


def test_missing_results_file_is_reported_as_missing(tmp_path):
    export_small(tmp_path)
    reports = validate.validate_job("A", tmp_path)
    assert all(r.missing for r in reports)
    assert [e["status"] for e in budget.read_ledger(tmp_path) if e["status"] != "exported"] == [
        "missing",
        "missing",
    ]


def test_zero_shot_job_validates_without_span_checks(tmp_path):
    frame = make_frame(60)
    parts = partitions_for(frame)
    parts = {**parts, "test": set(frame["id"])}
    chunks.export_job("D", tmp_path, frame, parts, chunk_size=30)
    tasks = sorted((tmp_path / "D" / "tasks").glob("chunk_*.jsonl"))
    results = tmp_path / "D" / "results"
    for task in tasks:
        ids = [json.loads(line)["id"] for line in task.read_text(encoding="utf-8").splitlines()]
        write_lines(
            results / task.name,
            [{"id": i, "label": "not_scam", "confidence": "high"} for i in ids],
        )
    reports = validate.validate_job("D", tmp_path)
    assert all(r.ok for r in reports)


def test_validate_without_export_fails_clearly(tmp_path):
    with pytest.raises(FileNotFoundError, match="export"):
        validate.validate_job("A", tmp_path)


# --- status ------------------------------------------------------------------------------------


def test_status_command_summarises_ledger(tmp_path, capsys):
    export_small(tmp_path)
    assert cli.main(["--root", str(tmp_path), "status", "A"]) == 0
    out = capsys.readouterr().out
    assert "exported chunks: 2" in out
    assert "merged rows: none" in out


def test_unicode_line_separators_do_not_break_line_counts(tmp_path):
    frame = make_frame(6)
    frame.loc[0, "text"] = "first second third\x85end"
    frame.loc[1, "text"] = "कृपया OTP न बताएं"
    parts = {"train": set(frame["id"]), "cal": set(), "test": set()}
    written = chunks.export_job("A", tmp_path, frame, parts, chunk_size=10)
    lines = written[0].read_text(encoding="utf-8").split("\n")
    rows = [json.loads(line) for line in lines if line]
    assert len(rows) == len(frame)
    assert {r["id"]: r["text"] for r in rows}[frame.loc[0, "id"]] == frame.loc[0, "text"]
