"""Runtime stub: KV/reset and real-token stats must survive Unicode/chunk boundaries."""

from types import SimpleNamespace

from rakshak.llm.ela_export import generate_pte


def test_resets_each_generation_stops_at_json_and_never_counts_characters():
    class Runner:
        resets = 0
        def reset(self):
            self.resets += 1
            self.stopped = False
        def stop(self):
            self.stopped = True
        def generate(self, prompt, config, token, stats):
            for piece in ('{"points":', '[]}', 'must not appear'):
                token(piece)
                if self.stopped:
                    break
            stats(SimpleNamespace(num_generated_tokens=7, num_prompt_tokens=11))
    runner = Runner()
    def config(**kwargs):
        return kwargs
    for _ in range(2):
        text, count, _, stats = generate_pte(runner, 'synthetic prompt', 40, config_factory=config)
        assert text == '{"points":[]}' and count == 7
        assert stats['num_prompt_tokens'] == 11
    assert runner.resets == 2


def test_retention_margin_cannot_be_replaced_by_passing_absolute_quality():
    from rakshak.llm.ela_export import retention_report

    def row(arm, i, passed):
        return {'arm': arm, 'sample_id': str(i), 'campaign_id': str(i), 'injected': False,
                'side': 'scam', 'family': None, 'points_pass_n': passed, 'points_n': 1,
                'unsupported_n': 0, 'safe_claim': False}
    base = [row(arm, i, 1) for arm in ('slot', 'slot_hinglish') for i in range(120)]
    pte = [row(arm, i, int(arm == 'slot_hinglish' or i >= 24))
           for arm in ('slot', 'slot_hinglish') for i in range(120)]
    report = retention_report(base, pte)
    assert report['absolute_and_safety']['decision'] == 'go'
    assert report['paired_point_pass']['slot']['ci95'][0] < -.05
    assert not report['retained']


def test_registration_requires_both_libraries_and_required_native_operators(tmp_path):
    import pytest

    from rakshak.llm.ela_export import register_runtime_ops

    llama, quant = tmp_path/'llama', tmp_path/'quant'
    llama.mkdir()
    quant.mkdir()
    (llama/'libcustom_ops_aot_lib.so').write_bytes(b'synthetic library fixture')
    loaded = []
    def registry():
        return ['llama::custom_sdpa.out', 'llama::update_cache.out',
                'quantized_decomposed::embedding_4bit.dtype_out']
    with pytest.raises(ValueError, match='AOT library'):
        register_runtime_ops(roots=(llama, quant), load_library=loaded.append,
                              get_registered=registry)
    (quant/'libquantized_ops_aot_lib.so').write_bytes(b'synthetic quant fixture')
    result = register_runtime_ops(roots=(llama, quant), load_library=loaded.append,
                                   get_registered=registry)
    assert all(result['required_operators'].values()) and len(result['loaded_libraries']) == 2
    with pytest.raises(ValueError, match='unavailable'):
        register_runtime_ops(roots=(llama, quant), load_library=loaded.append,
                              get_registered=lambda: [])
