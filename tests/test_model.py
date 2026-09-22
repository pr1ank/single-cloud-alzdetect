from backend.model_service import load_backbone


def test_missing_weights_are_explicit_demo_fallback(tmp_path, monkeypatch):
    def offline(*args, **kwargs):
        raise ConnectionError("Deliberate offline test")
    monkeypatch.setattr("backend.model_service.download", offline)
    model, info = load_backbone("cpu", tmp_path)
    assert not info["pretrained"]
    assert "NOT transfer learning" in info["warning"]
    assert not model.training
    assert all(not parameter.requires_grad for parameter in model.parameters())
