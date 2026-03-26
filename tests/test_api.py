"""Tests for FastAPI serving endpoints (mocked inference)."""

import io
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import src.serving.app as app_module


class DummyEngine:
    def __init__(self):
        self.model = True
        self.device = "cpu"

    def process_single_image(self, image, mask=None):
        return np.ones((64, 64), dtype=np.float32) * 0.5

    def process_video(self, frames, masks=None):
        return [np.ones((64, 64), dtype=np.float32) * 0.5 for _ in frames]


@pytest.fixture(autouse=True)
def mock_engine():
    app_module._engine = DummyEngine()
    yield
    app_module._engine = None


@pytest.fixture
def client():
    from fastapi.testclient import TestClient
    return TestClient(app_module.app)


def _make_png():
    buf = io.BytesIO()
    Image.new("RGB", (64, 64)).save(buf, format="PNG")
    return buf.getvalue()


def _make_mask_png():
    buf = io.BytesIO()
    Image.new("L", (64, 64), color=255).save(buf, format="PNG")
    return buf.getvalue()


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "healthy"


def test_matte_frame(client):
    resp = client.post(
        "/matte_frame",
        files=[
            ("image", ("img.png", _make_png(), "image/png")),
            ("mask", ("mask.png", _make_mask_png(), "image/png")),
        ],
    )
    assert resp.status_code == 200


def test_matte_frame_no_engine(client):
    app_module._engine = None
    resp = client.post(
        "/matte_frame",
        files=[
            ("image", ("img.png", _make_png(), "image/png")),
            ("mask", ("mask.png", _make_mask_png(), "image/png")),
        ],
    )
    assert resp.status_code == 503


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
