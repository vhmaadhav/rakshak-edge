"""Bounded LRU + TTL cache behind FrozenURLExpert. No ONNX model is loaded: a fake session returns
deterministic logits, and time is injected through a fake clock (no sleeping)."""

from __future__ import annotations

import threading

import numpy as np
import pytest

from rakshak.fusion import url_expert
from rakshak.fusion.url_expert import BoundedTTLCache, FrozenURLExpert
from rakshak.url_features import url_features


class FakeClock:
    def __init__(self, now=0.0):
        self.now = float(now)

    def __call__(self):
        return self.now


class FakeSession:
    """Logit is a fixed function of the raw bytes, so equal URLs always get equal scores."""

    def __init__(self):
        self.runs = 0

    def run(self, _outputs, feed):
        self.runs += 1
        raw = feed['raw'].astype(np.float64)
        logits = (raw.sum(axis=1) % 17) / 4.0 - 2.0
        return [logits.reshape(-1, 1).astype(np.float32)]


def make_expert(max_entries, ttl, clock):
    keys = sorted(url_features('https://example.com/a').keys())
    expert = object.__new__(FrozenURLExpert)
    expert.scaler = {'columns': keys, 'mean': np.zeros(len(keys)), 'std': np.ones(len(keys))}
    expert.session = FakeSession()
    expert.cache = BoundedTTLCache(max_entries, ttl, clock=clock)
    return expert


URLS = ['https://example.com/a', 'https://example.org/b', 'https://example.net/c/d']


def test_eviction_at_capacity_drops_least_recently_used():
    cache = BoundedTTLCache(max_entries=2, ttl_seconds=100, clock=FakeClock())
    cache.set('a', 0.1)
    cache.set('b', 0.2)
    cache.set('c', 0.3)  # over capacity: 'a' is the oldest
    assert len(cache) == 2
    assert cache.get('a') is None
    assert cache.get('b') == 0.2 and cache.get('c') == 0.3


def test_get_refreshes_recency_before_eviction():
    cache = BoundedTTLCache(max_entries=2, ttl_seconds=100, clock=FakeClock())
    cache.set('a', 0.1)
    cache.set('b', 0.2)
    assert cache.get('a') == 0.1  # 'b' is now the least recently used
    cache.set('c', 0.3)
    assert cache.get('b') is None
    assert cache.get('a') == 0.1 and cache.get('c') == 0.3


def test_ttl_expiry_uses_injected_clock():
    clock = FakeClock(0.0)
    cache = BoundedTTLCache(max_entries=8, ttl_seconds=10, clock=clock)
    cache.set('k', 0.5)
    clock.now = 9.9
    assert cache.get('k') == 0.5
    clock.now = 10.0  # expires at set-time + ttl
    assert cache.get('k') is None
    assert len(cache) == 0


def test_set_refreshes_expiry_of_existing_key():
    clock = FakeClock(0.0)
    cache = BoundedTTLCache(max_entries=8, ttl_seconds=10, clock=clock)
    cache.set('k', 0.5)
    clock.now = 8.0
    cache.set('k', 0.6)
    clock.now = 15.0  # 7 seconds after the second set
    assert cache.get('k') == 0.6


def test_clear_and_contains():
    cache = BoundedTTLCache(max_entries=8, ttl_seconds=10, clock=FakeClock())
    cache.set('k', 0.5)
    assert 'k' in cache
    cache.clear()
    assert 'k' not in cache and len(cache) == 0


def test_invalid_bounds_are_rejected():
    with pytest.raises(ValueError):
        BoundedTTLCache(max_entries=0)
    with pytest.raises(ValueError):
        BoundedTTLCache(ttl_seconds=0)


def test_defaults_match_documented_values():
    assert url_expert.DEFAULT_CACHE_MAX_ENTRIES == 4096
    assert url_expert.DEFAULT_CACHE_TTL_SECONDS == 3600.0


def test_hits_return_identical_scores_without_running_the_session():
    expert = make_expert(max_entries=16, ttl=100, clock=FakeClock())
    first = expert.score_urls(URLS)
    runs_after_first = expert.session.runs
    second = expert.score_urls(URLS)
    assert second == first
    assert expert.session.runs == runs_after_first  # every URL was a hit
    assert all(isinstance(v, float) for v in first)


def test_scores_do_not_depend_on_cache_size():
    big = make_expert(max_entries=16, ttl=100, clock=FakeClock())
    small = make_expert(max_entries=1, ttl=100, clock=FakeClock())
    assert small.score_urls(URLS) == big.score_urls(URLS)


def test_eviction_during_one_call_still_returns_every_score():
    expert = make_expert(max_entries=1, ttl=100, clock=FakeClock())
    scores = expert.score_urls(URLS)
    assert len(scores) == len(URLS)
    assert len(expert.cache) == 1


def test_expired_entries_are_recomputed_after_ttl():
    clock = FakeClock(0.0)
    expert = make_expert(max_entries=16, ttl=10, clock=clock)
    first = expert.score_urls(URLS)
    runs = expert.session.runs
    clock.now = 10.0
    again = expert.score_urls(URLS)
    assert expert.session.runs == runs + 1
    assert again == first


def test_score_aggregates_per_message_max_from_cache():
    expert = make_expert(max_entries=16, ttl=100, clock=FakeClock())
    texts = ['plain text', 'see https://example.com/a and https://example.org/b']
    scores, present = expert.score(texts)
    assert np.isnan(scores[0]) and present.tolist() == [False, True]
    expected = max(expert.score_urls(['https://example.com/a', 'https://example.org/b']))
    assert scores[1] == expected


def test_cache_is_safe_under_threads():
    cache = BoundedTTLCache(max_entries=32, ttl_seconds=100, clock=FakeClock())
    errors = []

    def worker(offset):
        try:
            for i in range(500):
                key = f'k{(i + offset) % 64}'
                cache.set(key, float(i))
                cache.get(key)
        except Exception as exc:  # pragma: no cover - surfaced below
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(n * 7,)) for n in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    assert len(cache) <= 32
