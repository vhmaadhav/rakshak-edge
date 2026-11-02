import numpy as np
import pytest

torch = pytest.importorskip('torch')

from rakshak.models.url_byt5 import URLByT5, encode_urls  # noqa: E402


def test_byte_ids_eos_padding_and_identical_200_byte_prefix():
    texts = ['', 'é', 'x'*199+'é']
    ids = encode_urls(texts)
    assert ids.shape == (3, 201) and ids.dtype == np.int64
    assert ids[0, 0] == 1 and not ids[0, 1:].any()
    for index, text in enumerate(texts):
        raw = text.encode('utf-8')[:200]
        assert ids[index, :len(raw)].tolist() == [value+3 for value in raw]
        assert ids[index, len(raw)] == 1 and not ids[index, len(raw)+1:].any()


def test_tiny_encoder_parameter_guards_masking_and_backward():
    from transformers import T5Config

    cfg = T5Config(vocab_size=384, d_model=16, d_kv=8, d_ff=32, num_layers=1,
                   num_heads=2, dropout_rate=0., eos_token_id=1, pad_token_id=0)
    torch.manual_seed(3)
    model = URLByT5(cfg, 2).eval()
    state = {key: value.clone() for key, value in model.encoder.state_dict().items()}
    model.load_encoder({**state, 'decoder.example': torch.zeros(1)})
    missing = dict(state)
    missing.pop(next(iter(missing)))
    with pytest.raises(ValueError, match='every exact'):
        model.load_encoder(missing)
    with pytest.raises(ValueError, match='unexpected'):
        model.load_encoder({**state, 'unexplained': torch.zeros(1)})
    ids = torch.tensor(encode_urls(['a', 'b']))
    side = torch.zeros(2, 2)
    logits = model(ids, side)
    torch.testing.assert_close(logits, model(ids[:, :4], side), rtol=1e-5, atol=1e-5)
    logits.sum().backward()
    assert any(param.grad is not None and param.grad.abs().sum() > 0
               for param in model.encoder.parameters())
    assert torch.isfinite(logits).all()
