"""Native cal membership matches the original carve; test URLs never enter the loader."""

import json
from pathlib import Path

import pandas as pd

from rakshak.eval.phreshphish import grouped_cal
from rakshak.fusion.native_url_cal import ids_hash, native_cal_rows


def test_native_url_loader_preserves_ids_and_excludes_test_urls(tmp_path, monkeypatch):
    rows = [{'id': f'train{i}', 'url': f'https://g{i//2}.example/{i}', 'label': i%2,
             'reg_domain': f'g{i//2}.example', 'url_norm': f'https://g{i//2}.example/{i}',
             'bench_full': False} for i in range(160)]
    rows.insert(13, {'id': 'test-only', 'url': 'https://never-score-test.invalid', 'label': 1,
                     'reg_domain': 'test.invalid', 'url_norm': 'test-only', 'bench_full': True})
    frame = pd.DataFrame(rows)
    frame.to_parquet(tmp_path/'urls_phreshphish.parquet')
    ids = [f'train{i}' for i in range(159, -1, -1)]
    split = tmp_path/'splits'
    split.mkdir()
    (split/'url_phreshphish.json').write_text(json.dumps({
        'partitions': {'official': {'train': ids, 'test': ['test-only']}}}))
    original = pd.Index(frame.id).get_indexer(ids)
    _, cal = grouped_cal(original, frame.label.to_numpy(),
                          frame.reg_domain.to_numpy()[original], .05, 0)
    expected = frame.iloc[cal].id.tolist()
    real_read, calls = pd.read_parquet, []
    def filtered_read(path, **kwargs):
        calls.append(kwargs)
        assert kwargs['filters'] == [('id', 'in', ids)]
        assert 'bench_full' not in kwargs['columns']
        result = real_read(path, **kwargs)
        assert 'test-only' not in set(result.id)
        assert not result.url.str.contains('never-score-test').any()
        return result
    monkeypatch.setattr(pd, 'read_parquet', filtered_read)
    selected, provenance = native_cal_rows(tmp_path)
    assert selected.id.tolist() == expected
    assert provenance['cal_ids'] == expected
    assert provenance['cal_ids_sha256'] == ids_hash(expected)
    assert provenance['benchmark_columns_loaded'] is False
    assert len(calls) == 1
    assert set(selected.label) == {0, 1}


def test_original_native_cal_algorithm_source_is_pinned():
    from rakshak.fusion.native_url_cal import CAL_SOURCE, sha256

    repo = Path(__file__).resolve().parents[1]
    assert sha256(repo/'src/rakshak/eval/phreshphish.py') == CAL_SOURCE


def test_native_cal_scores_bare_url_directly_without_sms_extractor(monkeypatch):
    from types import SimpleNamespace

    import numpy as np

    from rakshak.fusion import native_url_cal as module
    from rakshak.models.encoding import encode_batch

    values = ['bare-domain.invalid/path', 'not-an-sms-url', 'https://example.invalid']
    monkeypatch.setattr(module, 'url_features', lambda value: {'length': len(value)})
    calls = []
    class Session:
        def run(self, _, feed):
            calls.append(feed)
            assert np.array_equal(feed['raw'], encode_batch(values, 200).astype(np.int64))
            assert feed['side'].shape == (3, 1)
            return [np.asarray([0, 1, -1], dtype=np.float32)]
    url = SimpleNamespace(scaler={'columns': ['length'], 'mean': [0], 'std': [1]},
                          session=Session())
    scores = module.native_url_probabilities(url, values)
    assert len(scores) == 3 and len(calls) == 1
    assert scores[0] == .5 and scores[1] > .5 and scores[2] < .5
