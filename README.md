# Uranus SDK

Python SDK for the Uranus video-generation serving API.  Provides a
high-level `UranusSimulationSession` class that wraps HTTP communication,
response parsing (JSON and legacy multipart), session management, and
authentication.

## Installation

```bash
pip install uranus-sdk
```

## Quick Start (simplest)

Load a sample from HuggingFace, run inference, save to mp4:

```python
import os
import cv2
import numpy as np
from uranus_sdk import load_sample, UranusSimulationSession

os.environ["URANUS_BASE_URL"] = "https://uranus.d-robotics.cc/api/v1"
os.environ["URANUS_API_KEY"] = "sk-your-token"  # required by the hosted service

# 1. Load sample from HuggingFace Hub
#    (replace https://huggingface.co/ with https://hf-mirror.com/ to use hf-mirror)
sample = load_sample("https://huggingface.co/datasets/D-Robotics/Uranus-Demo-Data/resolve/main/rc-table30-v1__rc-aloha/000000",
                     num_chunks=8)

# 2. Create session + run step loop
with UranusSimulationSession(**sample["create"]) as session:
    all_frames = {}
    for step in sample["steps"]:
        frames = session.step(**step)
        for cam, imgs in frames.items():
            all_frames.setdefault(cam, []).extend(imgs)

# 3. Save to mp4
for cam, imgs in all_frames.items():
    h, w = imgs[0].shape[:2]
    writer = cv2.VideoWriter(f"{cam}.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 10, (w, h))
    for img in imgs:
        writer.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    writer.release()
```

## Configuration

Server URL and authentication are read from environment variables:

| Variable | Required | Default | Description |
|---|---|---|---|
| `URANUS_BASE_URL` | No | `https://uranus.d-robotics.cc/api/v1` | Server address |
| `URANUS_API_KEY` | Yes (hosted) | — | Bearer token; required by the hosted service, optional for local servers. When set, every request carries `Authorization: Bearer <token>` |

```bash
export URANUS_BASE_URL=https://uranus.d-robotics.cc/api/v1
export URANUS_API_KEY=sk-xxxx
```

## API

### `UranusSimulationSession(...)`

Construction triggers `POST /create` — the session is ready immediately.

**Parameters:**

All parameters are keyword-only.

| Parameter | Type | Required | Description |
|---|---|---|---|
| `prompt` | `str` | Yes | Natural-language task description |
| `mjcf_path` | `str \| Path \| None` | One of | MJCF XML file path (read and sent inline) |
| `mjcf_xml` | `str \| None` | One of | Inline MJCF XML; takes precedence over `mjcf_path`. `load_sample()` returns this form |
| `cameras` | `Sequence[CameraSpec]` | Yes | Ordered camera specs; order fixes output frame order. Each name must match a `<camera>` element in the MJCF |
| `ref_cam_images` | `Sequence[str \| Path \| bytes \| ndarray]` | Yes | One reference image per camera (path, bytes, or RGB array) |
| `ref_qpos` | `list[float] \| ndarray` | Yes | Complete MuJoCo qpos vector for the reference frame (length `model.nq`) |
| `robot2world_transform` | `ndarray \| None` | No | 4x4 robot-to-world matrix; omit for fixed-base (defaults to identity) |
| `end_effectors` | `Sequence[EESpec] \| None` | No | End-effector specs (aligned with OSS) |
| `skeleton` | `SkeletonSpec \| None` | No | Skeleton topology spec (aligned with OSS) |
| `target_size` | `tuple[int, int]` | No | Generation resolution `(height, width)`; default `(384, 640)` |
| `seed` | `int` | No | RNG seed; default `1` |
| `session_id` | `str \| None` | No | Custom session ID; auto-generated if omitted |
| `timeout` | `float` | No | Per-request HTTP timeout in seconds; default `600` |

Provide at least one of `mjcf_path` or `mjcf_xml`; if both are set, `mjcf_xml` wins.

### `session.step(qpos, *, robot2world_transforms=None)`

Generate video frames for a chunk of motion.

**Parameters:**

| Parameter | Type | Required | Description |
|---|---|---|---|
| `qpos` | `list[list[float]] \| ndarray` | Yes | Motion frames `(N, nq)`. Server rounds N up to a multiple of 4 |
| `robot2world_transforms` | `list[ndarray \| None] \| None` | No | Per-frame 4x4 transforms; None = identity |

**Returns:** `dict[str, list[np.ndarray]]` -- `{camera_name: [RGB uint8 (H,W,3), ...]}`

### `session.finish()`

Release the session. Idempotent. Called automatically by `__exit__`. Cleanup failures are ignored; if the create outcome is unknown, the server may keep the session record and return `409 cleanup_unconfirmed`.

### `load_sample(root, *, num_chunks=None, step_length=4)`

Load a sample from a local directory or an HTTP(S) URL (for example HuggingFace or hf-mirror) and return SDK-ready arguments.

| Parameter | Type | Required | Description |
|---|---|---|---|
| `root` | `str` | Yes | Local sample directory or HTTP(S) base URL |
| `num_chunks` | `int \| None` | No | Number of step chunks; `None` uses all complete chunks. Values above the available amount are clamped |
| `step_length` | `int` | No | Frames per chunk; default `4` |

Returns `{"create": dict, "steps": list[dict]}`. Pass `**sample["create"]` to `UranusSimulationSession` and `**step` to `session.step`.

## Spec Classes

The SDK ships self-contained Spec dataclasses and does not require the full `uranus` package. They serialize to the same JSON shape as `uranus.skeleton.specs`.

### `CameraSpec(name: str)`

`name` must match a `<camera name="...">` element in the MJCF; the server reads the camera pose and intrinsics from that element.

### `EESpec(object_type, object_name, radius_mode, pad_bodies=None, radius=None, sh_correction=None)`

- `object_type`: `"body"` or `"site"`.
- `radius_mode`: `"pad_pair"` (requires `pad_bodies`) or `"fixed"` (requires `radius` > 0).
- `sh_correction`: optional 3×3 rotation matrix.

### `SkeletonSpec(mode="full_tree", chains=(), skip_bodies=(), gripper_keypoint_overrides=())`

- `mode`: `"full_tree"` or `"chains"`; `chains` is required for `"chains"` mode.
- `gripper_keypoint_overrides`: optional `GripperKeypointOverride(ee_object_type, ee_object_name, finger_bodies, closing_axis=1)` entries.

## Error Handling

```python
from uranus_sdk import UranusSDKError

try:
    session = UranusSimulationSession(...)
except UranusSDKError as e:
    print(f"Failed: {e} (code={e.code}, status={e.status_code})")
```

Common server error codes; the SDK passes them through as `UranusSDKError.code`:

| Scenario | HTTP Status | Code |
|---|---|---|
| Invalid parameters | 400 | `invalid_payload` |
| Session not found | 404 | `session_not_found` |
| Session already exists | 409 | `session_exists` |
| Server busy | 503 | `server_busy` |
| Auth failure | 401 | `token_invalid` |

The server may return additional codes (for example `invalid_json`, `missing_field`, `unknown_session`, or `cleanup_unconfirmed`).

## License

This project is licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for the full text.
