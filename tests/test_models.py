"""Tests for the student models: byte encoding, rule side features, n-gram hashing, featurisation,
and (when torch is importable) forward shapes, dual fusion, size budgets, training, INT8, export.

Tiny random data only. The numpy/stdlib tests always run. Tests that need torch call
``pytest.importorskip("torch")`` so the local venv (no torch) skips them.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import load_fixture

from rakshak.canonicalize import canonicalize
from rakshak.models.encoding import (
    MAX_LEN,
    N_RULES,
    PAD_ID,
    VOCAB_SIZE,
    encode_batch,
    encode_bytes,
    ngram_ids,
    rule_flags,
)
from rakshak.models.features import Featurized, featurize
from rakshak.rules import EVIDENCE_TYPES, run_rules


def _torch():
    return pytest.importorskip("torch")


# ---------------------------------------------------------------- encoding (numpy only)


def test_devanagari_bytes_are_utf8():
    text = "नमस्ते"
    raw = text.encode("utf-8")
    ids = encode_bytes(text, MAX_LEN)
    assert len(raw) == 18  # six code points, three bytes each
    assert ids[: len(raw)].tolist() == list(raw)
    assert (ids[len(raw):] == PAD_ID).all()


def test_emoji_is_four_bytes():
    ids = encode_bytes("🙂", 8)
    assert ids[:4].tolist() == [240, 159, 153, 130]
    assert (ids[4:] == PAD_ID).all()


def test_truncation_and_empty_text():
    long_text = "a" * 1000
    ids = encode_bytes(long_text, 16)
    assert ids.shape == (16,) and (ids == ord("a")).all()
    empty = encode_bytes("", 16)
    assert (empty == PAD_ID).all()


def test_byte_ids_fit_vocabulary_and_batch_matches_single():
    texts = ["OTP दो", "बैंक KYC 🙂", ""]
    batch = encode_batch(texts, 32)
    assert batch.shape == (3, 32) and batch.max() < VOCAB_SIZE
    for row, text in enumerate(texts):
        assert batch[row].tolist() == encode_bytes(text, 32).tolist()


def test_rule_flags_match_run_rules_on_fixtures():
    cases = load_fixture("rule_cases.json")[:12]
    for case in cases:
        flags = rule_flags(canonicalize(case["text"]))
        assert flags.shape == (N_RULES,)
        fired = {EVIDENCE_TYPES[i] for i in np.flatnonzero(flags)}
        assert set(case["expect"]) <= fired, case["text"]
        expected = {e.type for e in run_rules(canonicalize(case["text"]))}
        assert fired == expected


def test_benign_text_has_no_rule_flags():
    flags = rule_flags(canonicalize("Thanks for your patience."))
    assert flags.sum() == 0


def test_ngram_ids_are_stable_and_bounded():
    buckets = 1024
    first = ngram_ids("Share your OTP to avoid suspension", buckets)
    second = ngram_ids("Share your OTP to avoid suspension", buckets)
    assert first.dtype == np.int32
    assert np.array_equal(first, second)
    assert first.min() >= 0 and first.max() < buckets
    assert ngram_ids("", buckets).tolist() == [0]


def test_featurize_shapes_and_views():
    texts = ["Share your OTP to avoid suspension", "Thanks for your patience.", "S B I KYC"]
    feats = featurize(texts)
    assert len(feats) == 3
    assert feats.raw_bytes.shape == (3, MAX_LEN) and feats.canon_bytes.shape == (3, MAX_LEN)
    assert feats.side_raw.shape == (3, N_RULES) and feats.side_canon.shape == (3, N_RULES)
    assert feats.raw_bytes.dtype == np.uint16
    for row, text in enumerate(texts):
        canon = canonicalize(text).normalized
        assert feats.canon_bytes[row].tolist() == encode_bytes(canon, MAX_LEN).tolist()
    assert feats.side("raw") is feats.side_raw and feats.side("dual") is feats.side_canon
    with pytest.raises(ValueError):
        feats.ngrams("raw")


def test_featurize_with_buckets_and_take_replace():
    texts = ["Share your OTP", "Thanks", "KYC blocked today"]
    feats = featurize(texts, buckets=512)
    assert len(feats.raw_ngrams) == 3 and len(feats.canon_ngrams) == 3
    sub = feats.take([2, 0])
    assert sub.texts == [texts[2], texts[0]]
    assert np.array_equal(sub.canon_bytes[0], feats.canon_bytes[2])
    repl = featurize(["Parcel customs fee pay now"], buckets=512)
    out = feats.replace([1], repl)
    assert out.texts == [texts[0], repl.texts[0], texts[2]]
    assert feats.texts == texts  # the original is untouched
    assert np.array_equal(out.raw_bytes[1], repl.raw_bytes[0])
    with pytest.raises(ValueError):
        feats.replace([0, 1], repl)


def test_featurized_empty_batch():
    feats = featurize([])
    assert len(feats) == 0 and feats.side_canon.shape == (0, N_RULES)
    assert isinstance(feats, Featurized)


# ---------------------------------------------------------------- torch models


def _tiny_texts():
    scam = ["Share your OTP to avoid account suspension", "KYC blocked, update at sbi-kyc.in",
            "Install AnyDesk QuickSupport to fix KYC", "Your parcel is held, pay customs fee"]
    ham = ["Thanks for your patience.", "Happy Diwali from all of us!", "Meeting at 5 pm",
           "Your OTP is 4821. Do not share it with anyone."]
    texts = scam * 6 + ham * 6
    labels = np.array([1] * len(scam) * 6 + [0] * len(ham) * 6, dtype=np.int64)
    return texts, labels


def test_charcnn_forward_shapes_single_and_dual():
    torch = _torch()
    from rakshak.models.charcnn import BASE, CharCNN

    feats = featurize(["Share your OTP", "Thanks", "नमस्ते 🙂"])
    single = CharCNN(BASE, views=("canon",), n_side=N_RULES).eval()
    views = {"canon": torch.as_tensor(feats.canon_bytes.astype(np.int64))}
    side = torch.as_tensor(feats.side_canon)
    with torch.no_grad():
        out = single(views, side)
    assert out.shape == (3,) and torch.isfinite(out).all()

    dual = CharCNN(BASE, views=("raw", "canon"), n_side=N_RULES).eval()
    views2 = {"raw": torch.as_tensor(feats.raw_bytes.astype(np.int64)),
              "canon": torch.as_tensor(feats.canon_bytes.astype(np.int64))}
    with torch.no_grad():
        out2 = dual(views2, side)
    assert out2.shape == (3,) and torch.isfinite(out2).all()


def test_dual_fusion_depends_on_both_views_and_side():
    torch = _torch()
    from rakshak.models.charcnn import BASE, CharCNN

    torch.manual_seed(0)
    model = CharCNN(BASE, views=("raw", "canon"), n_side=N_RULES).eval()
    feats = featurize(["Share your OTP to avoid suspension"])
    raw = torch.as_tensor(feats.raw_bytes.astype(np.int64))
    canon = torch.as_tensor(feats.canon_bytes.astype(np.int64))
    side = torch.as_tensor(feats.side_canon)
    other_raw = torch.as_tensor(featurize(["Thanks for your patience."]).raw_bytes.astype(np.int64))
    with torch.no_grad():
        base = model({"raw": raw, "canon": canon}, side)
        raw_changed = model({"raw": other_raw, "canon": canon}, side)
        canon_changed = model({"raw": raw, "canon": other_raw}, side)
        side_changed = model({"raw": raw, "canon": canon}, torch.ones_like(side))
    assert not torch.allclose(base, raw_changed)
    assert not torch.allclose(base, canon_changed)
    assert not torch.allclose(base, side_changed)
    with pytest.raises(ValueError):
        model({"raw": raw, "canon": canon}, None)


def test_all_padding_row_is_finite():
    torch = _torch()
    from rakshak.models.charcnn import BASE, CharCNN

    model = CharCNN(BASE, views=("canon",), n_side=0).eval()
    pad = torch.full((2, MAX_LEN), PAD_ID, dtype=torch.long)
    with torch.no_grad():
        out = model({"canon": pad})
    assert out.shape == (2,) and torch.isfinite(out).all()


def test_size_budgets_and_size_accounting():
    _torch()
    from rakshak.models.student import ModelSpec, build_model, param_count, state_bytes

    limits = {("base", "charcnn"): 2e6, ("base", "dual"): 2e6,
              ("wide", "charcnn"): 10e6, ("wide", "dual"): 10e6}
    for (size, kind), limit in limits.items():
        variant = "dual" if kind == "dual" else "canon"
        model = build_model(ModelSpec(kind, size, variant))
        fp32 = 4 * param_count(model)
        on_disk = state_bytes(model)
        assert fp32 < limit, (size, kind, fp32)
        assert fp32 <= on_disk < fp32 + 20_000, (size, kind, fp32, on_disk)


def test_model_spec_rejects_bad_combinations():
    _torch()
    from rakshak.models.student import ModelSpec

    with pytest.raises(ValueError):
        ModelSpec("dual", "base", "raw")
    with pytest.raises(ValueError):
        ModelSpec("charcnn", "base", "dual")
    with pytest.raises(ValueError):
        ModelSpec("fasttext", None, "dual")
    with pytest.raises(ValueError):
        ModelSpec("charcnn", "huge", "raw")


def test_fasttext_pack_bags_and_forward():
    torch = _torch()
    from rakshak.models.fasttext import HashedFastText, pack_bags

    indices, offsets = pack_bags([np.array([1, 2]), np.array([5]), np.array([7, 8, 9])])
    assert indices.tolist() == [1, 2, 5, 7, 8, 9]
    assert offsets.tolist() == [0, 2, 3]
    model = HashedFastText(buckets=64, dim=8).eval()
    with torch.no_grad():
        out = model(torch.as_tensor(indices), torch.as_tensor(offsets))
    assert out.shape == (3,) and torch.isfinite(out).all()


def test_train_quantize_and_checkpoint_roundtrip(tmp_path):
    torch = _torch()
    from rakshak.models.student import (
        ModelSpec,
        TrainConfig,
        load_checkpoint,
        predict_scores,
        quantize_int8,
        save_checkpoint,
        train_student,
    )

    texts, labels = _tiny_texts()
    feats = featurize(texts, buckets=256)
    train_idx = np.arange(0, len(texts), 2)
    cal_idx = np.arange(1, len(texts), 2)
    train, cal = feats.take(train_idx), feats.take(cal_idx)
    for spec in (ModelSpec("charcnn", "base", "canon", True),
                 ModelSpec("dual", "base", "dual", False),
                 ModelSpec("fasttext", None, "raw", True, buckets=256, dim=8)):
        cfg = TrainConfig(epochs=2, batch_size=16, adv_p=0.5, seed=1)
        model, info = train_student(spec, train, labels[train_idx], cal, labels[cal_idx],
                                    cfg, device=torch.device("cpu"))
        assert info["epochs_run"] == 2 and len(info["history"]) == 2
        scores = predict_scores(model, spec, cal, "cpu")
        assert scores.shape == (len(cal_idx),) and np.all((scores >= 0) & (scores <= 1))
        qmodel, notes = quantize_int8(model, spec)
        qscores = predict_scores(qmodel, spec, cal, "cpu")
        assert qscores.shape == scores.shape and np.all(np.isfinite(qscores))
        assert notes
        path = tmp_path / f"{spec.label}.pt"
        save_checkpoint(path, model, spec, 0.5, extra={"k": 1})
        loaded, spec2, thr = load_checkpoint(path)
        assert spec2 == spec and thr == 0.5
        assert np.allclose(predict_scores(loaded, spec2, cal, "cpu"), scores)


def test_distillation_path_runs_with_unlabeled_rows_and_missing_teacher():
    torch = _torch()
    from rakshak.models.student import ModelSpec, TrainConfig, train_student

    texts, labels = _tiny_texts()
    feats = featurize(texts)
    train_idx = np.arange(0, len(texts), 2)
    cal_idx = np.arange(1, len(texts), 2)
    # Teacher logits [logit_0, logit_1] aligned with the training rows; every third row unknown.
    teacher = np.stack([np.where(labels[train_idx] == 1, -2.0, 2.0),
                        np.where(labels[train_idx] == 1, 2.0, -2.0)], axis=1)
    teacher[np.arange(len(train_idx)) % 3 == 0] = np.nan
    # Append unlabeled rows (NaN label): they get the KD term only.
    extra = np.arange(1, 9, 2)
    rows = np.concatenate([train_idx, extra])
    labels_pool = np.concatenate([labels[train_idx].astype(float), np.full(len(extra), np.nan)])
    teacher_pool = np.concatenate([teacher, np.stack([[0.5, -0.5]] * len(extra))])
    spec = ModelSpec("charcnn", "base", "canon", False)
    cfg = TrainConfig(epochs=2, batch_size=16, seed=2)
    model, info = train_student(spec, featurize(texts).take(rows), labels_pool,
                                feats.take(cal_idx), labels[cal_idx], cfg,
                                device=torch.device("cpu"), teacher=teacher_pool)
    assert np.isfinite([h["train_loss"] for h in info["history"]]).all()
    assert info["rows"]["labelled"] == len(train_idx)
    assert info["rows"]["unlabeled"] == len(extra)
    with pytest.raises(ValueError):
        train_student(spec, feats.take(train_idx), labels[train_idx].astype(float),
                      feats.take(cal_idx), labels[cal_idx], cfg, device=torch.device("cpu"),
                      teacher=teacher[:-1])


def test_distillation_loss_terms_and_sign():
    torch = _torch()
    from rakshak.models.student import distillation_loss

    pos_w = torch.tensor(1.0)
    nan = float("nan")
    # Teacher margin logit_1 - logit_0 = 4 for the first row, -4 for the second.
    teacher = torch.tensor([[0.0, 4.0], [0.0, -4.0]])
    alpha, temp = 0.7, 2.0

    # KL is zero when the student margin equals the teacher margin; unlabeled rows contribute
    # only the KD term, so the loss is exactly zero.
    matched = torch.tensor([4.0, -4.0])
    unlabeled = torch.tensor([nan, nan])
    loss0 = distillation_loss(matched, unlabeled, teacher, pos_weight=pos_w, alpha=alpha,
                              temp=temp)
    assert float(loss0) == pytest.approx(0.0, abs=1e-6)

    # Labelled rows with a teacher: (1 - alpha) BCE + alpha T^2 KL, checked against the formula.
    logits = torch.tensor([1.0, 0.5])
    labels = torch.tensor([1.0, 0.0])
    got = distillation_loss(logits, labels, teacher, pos_weight=pos_w, alpha=alpha, temp=temp)
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    p_t = torch.softmax(teacher / temp, dim=1)
    p_s = torch.softmax(torch.stack([torch.zeros(2), logits], dim=1) / temp, dim=1)
    kl = (p_t * (p_t.log() - p_s.log())).sum(dim=1)
    want = ((1 - alpha) * bce + alpha * temp * temp * kl).mean()
    assert torch.isclose(got, want, atol=1e-5)
    assert float(kl.min()) >= 0.0

    # Labelled rows without a teacher get plain BCE with pos_weight.
    no_teacher = torch.full((2, 2), nan)
    plain = distillation_loss(logits, labels, no_teacher, pos_weight=torch.tensor(2.0),
                              alpha=alpha, temp=temp)
    want_plain = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, labels, pos_weight=torch.tensor(2.0))
    assert torch.isclose(plain, want_plain, atol=1e-6)

    # Sign: the KD gradient pushes a too-high student margin down and a too-low one up.
    z = torch.tensor([6.0, -6.0], requires_grad=True)
    kd_only = distillation_loss(z, unlabeled, teacher, pos_weight=pos_w, alpha=1.0, temp=temp)
    kd_only.backward()
    assert z.grad[0] > 0 and z.grad[1] < 0

    # Unlabeled rows with no teacher contribute nothing.
    assert float(distillation_loss(logits[:1], unlabeled[:1], no_teacher[:1], pos_weight=pos_w,
                                   alpha=alpha, temp=temp)) == 0.0


def test_onnx_export_and_checkpoint_parity(tmp_path):
    _torch()
    pytest.importorskip("onnx")
    from rakshak.models.student import ModelSpec, build_model, export_onnx, onnx_parity

    spec = ModelSpec("dual", "base", "dual", True)
    model = build_model(spec).eval()
    feats = featurize(["Share your OTP to avoid suspension", "Thanks"])
    path = tmp_path / "student.onnx"
    info = export_onnx(model, path)
    assert path.stat().st_size > 0 and info["inputs"] == ["raw", "canon", "side"]
    import onnx

    onnx.checker.check_model(str(path))
    parity = onnx_parity(model, path, feats, n=2)
    if "max_abs_logit_diff" in parity:
        assert parity["max_abs_logit_diff"] < 1e-3


def test_latency_helper_reports_percentiles():
    _torch()
    from rakshak.models.student import latency_ms

    stats = latency_ms(lambda t: sum(ord(c) for c in t), ["abc", "defg"], n=20)
    assert stats["n"] == 20 and stats["threads"] == 1
    assert stats["p95_ms"] >= stats["p50_ms"] >= 0.0
