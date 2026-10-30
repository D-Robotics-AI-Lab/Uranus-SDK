"""Unit tests for response decoding (JSON serving-v2 and legacy multipart)."""

from __future__ import annotations

import base64
import json

import cv2
import numpy as np
import requests

from uranus_sdk.session import _decode_json, _decode_response


def _response(body: bytes, content_type: str) -> requests.Response:
    response = requests.Response()
    response.status_code = 200
    response.headers["Content-Type"] = content_type
    response._content = body
    return response


def _png_b64(height: int, width: int) -> str:
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:, :, 1] = 200
    ok, buffer = cv2.imencode(".png", image)
    assert ok
    return base64.b64encode(buffer.tobytes()).decode("ascii")


def test_decode_json_step_response() -> None:
    body = {
        "session_id": "s1",
        "output": {
            "obs": {"head_0": [_png_b64(8, 12), _png_b64(8, 12)]},
            "num_step": 2,
        },
    }
    info, frames = _decode_json(
        _response(json.dumps(body).encode(), "application/json")
    )

    assert info["session_id"] == "s1"
    assert list(frames) == ["head_0"]
    assert len(frames["head_0"]) == 2
    assert frames["head_0"][0].shape == (8, 12, 3)


def test_decode_response_detects_json_without_content_type() -> None:
    body = {"output": {"obs": {"side": [_png_b64(4, 6)]}}}
    info, frames = _decode_response(
        _response(json.dumps(body).encode(), "text/plain")
    )

    assert "output" in info
    assert frames["side"][0].shape == (4, 6, 3)


def test_decode_response_detects_multipart() -> None:
    payload = b'{"ok": true}\r\n'
    body = (
        b"--boundary\r\n"
        b"Content-Type: application/json\r\n\r\n" + payload + b"\r\n"
        b"--boundary\r\n"
        b"Content-Type: image/jpeg\r\n"
        b"X-Camera: wrist\r\n\r\n" + b"\xff\xd8\xff\xd9" + b"\r\n"
        b"--boundary--\r\n"
    )
    info, frames = _decode_response(
        _response(body, "multipart/mixed; boundary=boundary")
    )

    assert info == {"ok": True}
    assert "wrist" in frames
