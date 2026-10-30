"""Tests for the dataset pipeline on tiny in-memory fixtures (no raw data needed)."""

from __future__ import annotations

import hashlib

import pandas as pd
import pytest

from rakshak.datasets.common import dedup_key, stable_id
from rakshak.datasets.dedup import (
    assign_campaigns,
    mark_exact_duplicates,
    minhash_signatures,
    near_duplicate_pairs,
)
from rakshak.datasets.lang import detect_lang
from rakshak.datasets.loaders import (
    SMS_RAW_COLUMNS,
    SOURCE_ANMOL,
    SOURCE_IMC25,
    SOURCE_SCAMSHIELD,
    SOURCE_UCI,
)
from rakshak.datasets.sms import build_sms_frame
from rakshak.datasets.splits import (
    assign_group_splits,
    campaign_partition,
    check_disjoint,
    lang_shift_partition,
    temporal_partition,
    url_domain_partition,
)
from rakshak.datasets.template import template_key
from rakshak.datasets.unionfind import UnionFind
from rakshak.datasets.urls import build_url_frame, normalize_url


def _raw(rows: list[dict]) -> pd.DataFrame:
    base = {c: None for c in SMS_RAW_COLUMNS}
    base.update(
        source=SOURCE_UCI,
        subsource="",
        source_split=None,
        date=None,
        synthetic=None,
        augmentation_only=False,
        src_lang=None,
    )
    frame = pd.DataFrame([{**base, **r} for r in rows], columns=SMS_RAW_COLUMNS)
    frame["synthetic"] = pd.array(frame["synthetic"].tolist(), dtype="boolean")
    return frame


# ---------------------------------------------------------------------------
# ids, keys, template, language
# ---------------------------------------------------------------------------


def test_stable_id_is_deterministic_and_source_sensitive():
    assert stable_id("uci", "00001") == stable_id("uci", "00001")
    assert stable_id("uci", "00001") != stable_id("mishra", "00001")
    assert len(stable_id("uci", "1")) == 20
    expected = hashlib.sha1(b"uci\x1f1").hexdigest()[:20]
    assert stable_id("uci", "1") == expected


def test_dedup_key_collapses_case_and_whitespace():
    assert dedup_key("  Hello   WORLD\n\tok ") == "hello world ok"


def test_template_key_masks_urls_numbers_amounts_phones_keeps_months():
    msg = (
        "KYC expires 16-sep. Visit http://bad.example.xyz/x?id=5 or call "
        "+91 98765 43210, pay Rs 500"
    )
    key = template_key(msg)
    for token in ("<URL>", "<PHONE>", "<AMT>", "<NUM>"):
        assert token in key
    assert "sep" in key
    assert not any(ch.isdigit() for ch in key)


def test_template_key_equal_for_messages_differing_only_in_slots():
    a = "Your SBI account is blocked. Click http://a1.example.xyz/ref?x=1 now, pay Rs 500"
    b = "Your SBI account is blocked. Click http://zz9.example.top/ref?x=77 now, pay Rs 9000"
    assert template_key(a) == template_key(b)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("आपका KYC अधूरा है, जल्दी करें", "hi_deva"),
        ("aapka KYC pending hai, jaldi karo", "hinglish"),
        ("Your account is blocked, click the link to verify", "en"),
        ("Tentamos entregar a sua encomenda hoje", "other"),
        ("", "other"),
    ],
)
def test_detect_lang_buckets(text, expected):
    assert detect_lang(text) == expected


# ---------------------------------------------------------------------------
# union-find, exact dedup, near duplicates, campaigns
# ---------------------------------------------------------------------------


def test_union_find_components_and_labels():
    uf = UnionFind(6)
    uf.union(0, 1)
    uf.union(1, 2)
    uf.union(4, 5)
    labels = uf.labels()
    assert labels[0] == labels[1] == labels[2]
    assert labels[4] == labels[5]
    assert len({labels[0], labels[3], labels[4]}) == 3


def test_exact_dedup_keeps_first_and_flags_label_conflicts():
    df = pd.DataFrame(
        {
            "id": ["a", "b", "c", "d"],
            "dedup_key": ["x", "x", "y", "y"],
            "label_binary": [0, 0, 1, 0],
        }
    )
    out = mark_exact_duplicates(df)
    assert out.loc[0, "dup_of"] != out.loc[0, "dup_of"]  # NaN: representative
    assert out.loc[1, "dup_of"] == "a"
    assert out.loc[2, "dup_of"] != out.loc[2, "dup_of"]
    assert out.loc[3, "dup_of"] == "c"
    assert out["label_conflict"].tolist() == [False, False, True, True]


def test_near_duplicate_pairs_threshold():
    base = "your kyc is pending update now at the official portal to avoid blocking "
    templates = [
        base * 2,
        base * 2 + "today",
        "lunch at the canteen was good and the weather stayed warm all afternoon long",
    ]
    pairs = near_duplicate_pairs(minhash_signatures(templates), 0.7)
    assert [(i, j) for i, j, _ in pairs] == [(0, 1)]
    assert all(est >= 0.7 for _, _, est in pairs)


def test_campaigns_join_near_duplicates_and_shared_scam_domains_only():
    raw = _raw(
        [
            {
                "source": SOURCE_UCI,
                "row_key": "1",
                "text": "Win a prize at http://evil-a.xyz/p1",
                "label_binary": 1,
                "label_fine": "spam",
            },
            {
                "source": SOURCE_UCI,
                "row_key": "2",
                "text": "Claim your prize at http://evil-b.xyz/z",
                "label_binary": 1,
                "label_fine": "spam",
            },
            {
                "source": SOURCE_UCI,
                "row_key": "3",
                "text": "Ok see you at 5 then",
                "label_binary": 0,
                "label_fine": "ham",
            },
            {
                "source": SOURCE_UCI,
                "row_key": "4",
                "text": "Ok see you at 6 then",
                "label_binary": 0,
                "label_fine": "ham",
            },
            {
                "source": SOURCE_UCI,
                "row_key": "5",
                "text": "Lunch at noon?",
                "label_binary": 0,
                "label_fine": "ham",
            },
            {
                "source": SOURCE_UCI,
                "row_key": "6",
                "text": "Lunch at noon? see",
                "label_binary": 0,
                "label_fine": "ham",
            },
            {
                "source": SOURCE_UCI,
                "row_key": "7",
                "text": "Shortened link http://bit.ly/abc1",
                "label_binary": 1,
                "label_fine": "spam",
            },
            {
                "source": SOURCE_UCI,
                "row_key": "8",
                "text": "Other shortened http://bit.ly/zzz9",
                "label_binary": 1,
                "label_fine": "spam",
            },
        ]
    )
    frame, diag = build_sms_frame(raw)
    camp = frame.set_index("text")["campaign_id"]
    # The two prize messages share no template but share a non-shortener registered domain?
    # They do not (evil-a.xyz vs evil-b.xyz are different domains), so they stay apart.
    assert (
        camp["Win a prize at http://evil-a.xyz/p1"]
        != camp["Claim your prize at http://evil-b.xyz/z"]
    )
    # Ham near-duplicates (differ by one digit in template terms: same after masking) join.
    assert camp["Ok see you at 5 then"] == camp["Ok see you at 6 then"]
    # Shortener domains never join unrelated scam messages.
    assert camp["Shortened link http://bit.ly/abc1"] != camp["Other shortened http://bit.ly/zzz9"]
    assert diag["n_campaigns"] >= 4


def test_shared_scam_domain_unions_campaign_members():
    df = pd.DataFrame(
        {
            "id": ["p", "q", "r"],
            "template_key": [
                "alpha beta gamma delta one",
                "zeta eta theta iota kappa lambda mu",
                "nu xi omicron pi rho sigma tau",
            ],
            "label_binary": [1, 1, 0],
            "reg_domains": [["shared-evil.xyz"], ["shared-evil.xyz"], ["other.com"]],
            "dup_of": [None, None, None],
        }
    )
    out, _ = assign_campaigns(df)
    camp = dict(zip(out["id"], out["campaign_id"], strict=True))
    assert camp["p"] == camp["q"]
    assert camp["r"] != camp["p"]


def test_duplicates_inherit_representative_campaign():
    raw = _raw(
        [
            {
                "source": SOURCE_UCI,
                "row_key": "1",
                "text": "Hello there friend",
                "label_binary": 0,
                "label_fine": "ham",
            },
            {
                "source": SOURCE_SCAMSHIELD,
                "row_key": "real_9",
                "text": "hello  THERE friend",
                "label_binary": 0,
                "label_fine": "ham",
                "synthetic": False,
            },
        ]
    )
    frame, _ = build_sms_frame(raw)
    assert frame["dup_of"].notna().sum() == 1
    assert frame["campaign_id"].notna().all()
    assert frame["campaign_id"].nunique() == 1


# ---------------------------------------------------------------------------
# eval eligibility and split invariants
# ---------------------------------------------------------------------------


def _fixture_frame(n_groups: int = 60) -> pd.DataFrame:
    rows = []
    for g in range(n_groups):
        label = g % 2
        src = [SOURCE_UCI, SOURCE_SCAMSHIELD, SOURCE_IMC25][g % 3]
        for k in range(3):
            synthetic = (
                True if (g % 17 == 0 and k == 0) else (False if src == SOURCE_SCAMSHIELD else None)
            )
            rows.append(
                {
                    "source": src,
                    "row_key": f"{g}-{k}",
                    "text": f"group {g} lure number {k} about account {g * 7} for alert {label}",
                    "label_binary": label,
                    "label_fine": "scam" if label else "ham",
                    "source_split": "train",
                    "date": f"2023-{1 + (g % 12):02d}-{1 + k:02d}" if src == SOURCE_IMC25 else None,
                    "synthetic": synthetic,
                    "augmentation_only": src == SOURCE_ANMOL,
                }
            )
    rows.append(
        {
            "source": SOURCE_ANMOL,
            "row_key": "aug",
            "text": "augmentation only row",
            "label_binary": 1,
            "label_fine": "spam",
            "augmentation_only": True,
        }
    )
    frame, _ = build_sms_frame(_raw(rows))
    return frame


def test_campaign_split_is_group_disjoint_and_clean():
    frame = _fixture_frame()
    kept = frame[frame["dup_of"].isna()]
    parts, _ = campaign_partition(kept)
    assert set(parts) == {"train", "cal", "test"}
    overlap = check_disjoint(kept, parts, "campaign_id")
    assert all(v == 0 for v in overlap.values()), overlap
    by_id = kept.set_index("id")
    for split in ("cal", "test"):
        sub = by_id.loc[parts[split]]
        assert not sub["synthetic"].fillna(False).astype(bool).any()
        assert sub["eval_ok"].all()
    # Every row of a group that holds a synthetic or augmentation-only row stays in train.
    bad = kept[kept["synthetic"].fillna(False).astype(bool) | ~kept["eval_ok"].astype(bool)]
    assert set(bad["id"]) <= set(parts["train"])


def test_temporal_split_uses_dates_and_pairs_ham_from_campaign_split():
    frame = _fixture_frame()
    kept = frame[frame["dup_of"].isna()]
    _, camp_split = campaign_partition(kept)
    parts = temporal_partition(kept, camp_split)
    by_id = kept.set_index("id")
    scam_test = [i for i in parts["test"] if by_id.loc[i, "label_binary"] == 1]
    scam_train = [i for i in parts["train"] if by_id.loc[i, "label_binary"] == 1]
    if scam_test and scam_train:
        assert by_id.loc[scam_train, "date"].max() <= by_id.loc[scam_test, "date"].min()
    ham_test = [i for i in parts["test"] if by_id.loc[i, "label_binary"] == 0]
    assert all(by_id.loc[i, "eval_ok"] for i in ham_test)


def test_lang_shift_trains_on_english_and_tests_on_hindi_or_hinglish():
    rows = []
    for g in range(40):
        lang_text = (
            "aapka account band hai jaldi karo"
            if g % 4 == 0
            else ("your account is blocked click the link now")
        )
        rows.append(
            {
                "source": SOURCE_UCI,
                "row_key": f"l{g}",
                "text": f"{lang_text} {g}",
                "label_binary": 1 if g % 4 == 0 else 0,
                "label_fine": "scam" if g % 4 == 0 else "ham",
            }
        )
    frame, _ = build_sms_frame(_raw(rows))
    kept = frame[frame["dup_of"].isna()]
    _, camp_split = campaign_partition(kept)
    parts = lang_shift_partition(kept, camp_split)
    by_id = kept.set_index("id")
    assert set(by_id.loc[parts["train"], "lang"]) <= {"en"}
    assert set(by_id.loc[parts["test"], "lang"]) <= {"hinglish", "hi_deva"}


def test_assign_group_splits_keeps_groups_whole_and_forces_flagged_to_train():
    df = pd.DataFrame(
        {
            "g": ["a", "a", "b", "b", "c", "c", "d", "e", "f"],
            "label": [0, 1, 0, 0, 1, 1, 0, 1, 0],
            "src": ["s1", "s1", "s2", "s2", "s1", "s1", "s2", "s2", "s1"],
            "forced": [False, False, False, False, True, False, False, False, False],
        }
    )
    split = assign_group_splits(
        df,
        "g",
        {"train": 0.6, "cal": 0.2, "test": 0.2},
        ["label", "src"],
        forced_col="forced",
        seed=3,
    )
    assert split[df["g"] == "a"].nunique() == 1
    assert split[df["g"] == "c"].tolist() == ["train", "train"]
    assert set(split) <= {"train", "cal", "test"}


def test_source_identical_partitions_are_deterministic():
    frame = _fixture_frame()
    kept = frame[frame["dup_of"].isna()]
    a, _ = campaign_partition(kept)
    b, _ = campaign_partition(kept)
    assert a == b


# ---------------------------------------------------------------------------
# URL corpus
# ---------------------------------------------------------------------------


def test_normalize_url_canonicalises_case_www_and_fragment():
    assert normalize_url("HTTPS://WWW.Example.com/") == "example.com"
    assert normalize_url("http://example.com:80/a?b=1#frag") == "example.com/a?b=1"
    assert normalize_url("https://example.com/a") == normalize_url("http://example.com/a")
    assert normalize_url("example.com/login") == "example.com/login"


def _url_raw(rows):
    cols = ["source", "row_key", "url", "label", "date", "brand"]
    return pd.DataFrame([{**{c: None for c in cols}, **r} for r in rows], columns=cols)


def test_url_dedup_drops_conflicts_and_keeps_first_by_priority():
    raw = _url_raw(
        [
            {
                "source": "phishtank_online_valid",
                "row_key": "1",
                "url": "https://a.example.xyz/",
                "label": 1,
                "date": "2024-01-01",
            },
            {
                "source": "legitphish_mendeley",
                "row_key": "2",
                "url": "http://A.example.xyz",
                "label": 0,
            },
            {
                "source": "phreshphish",
                "row_key": "3",
                "url": "https://b.example.com/x",
                "label": 0,
                "date": "2025-01-01",
            },
            {
                "source": "legitphish_mendeley",
                "row_key": "4",
                "url": "https://b.example.com/x",
                "label": 0,
            },
            {
                "source": "openphish_community_feed",
                "row_key": "5",
                "url": "https://c.example.net",
                "label": 1,
            },
        ]
    )
    kept, accounting = build_url_frame(raw)
    assert "https://a.example.xyz/" not in kept["url"].tolist()  # conflicting -> dropped
    assert set(kept["url"]) == {"https://b.example.com/x", "https://c.example.net"}
    assert accounting["legitphish_mendeley"]["label_conflict_dropped"] == 1
    assert accounting["phreshphish"]["kept"] == 1


def test_url_domain_split_never_shares_a_registered_domain():
    rows = []
    for i in range(50):
        rows.append(
            {
                "source": "phreshphish",
                "row_key": f"s{i}",
                "url": f"https://site{i % 10}.example{i % 5}.com/p{i}",
                "label": i % 2,
                "date": f"2025-01-{1 + i % 28:02d}",
            }
        )
    kept, _ = build_url_frame(_url_raw(rows))
    kept["group"] = kept["reg_domain"].where(kept["reg_domain"] != "", kept["url_norm"])
    parts, _ = url_domain_partition(kept)
    overlap = check_disjoint(kept, parts, "reg_domain")
    assert all(v == 0 for v in overlap.values()), overlap
