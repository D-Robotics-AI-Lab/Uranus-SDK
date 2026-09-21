# Uranus SDK

Python SDK for the Uranus video-generation serving API.  Provides a
high-level `UranusSimulationSession` class that wraps HTTP communication,
multipart parsing, session management, and authentication.

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

os.environ["URANUS_BASE_URL"] = "http://localhost:8000"
os.environ["URANUS_API_KEY"] = "sk-your-token"  # optional

# 1. Load sample from HuggingFace Hub (set HF_ENDPOINT to use a mirror)
sample = load_sample("D-Robotics/Uranus-Demo-Data/resolve/main/rc-table30-v1__rc-aloha/000000",
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

| Variable | Required | Default | Description |
|---|---|---|---|
| `URANUS_BASE_URL` | No | `http://localhost:8000` | Server address |
| `URANUS_API_KEY` | No | -- | Bearer token |
| `HF_ENDPOINT` | No | `https://huggingface.co` | HuggingFace endpoint (use `https://hf-mirror.com` for mirror) |

## Configuration

Server URL and authentication are read from environment variables:

| Variable | Required | Default | Description |
|---|---|---|---|
| `URANUS_BASE_URL` | No | `http://localhost:8000` | Server address |
| `URANUS_API_KEY` | No | — | Bearer token; when set, every request carries `Authorization: Bearer <token>` |

```bash
export URANUS_BASE_URL=http://uranus-server:8000
export URANUS_API_KEY=sk-xxxx
```

## API

### `UranusSimulationSession(...)`

Construction triggers `POST /create` — the session is ready immediately.

**Parameters:**

| Parameter | Type | Required | Description |
|---|---|---|---|
| `prompt` | `str` | Yes | Natural-language task description |
| `mjcf_path` | `str \| Path` | Yes | MJCF XML file path (read and sent inline) |
| `cameras` | `list[CameraSpec]` | Yes | Ordered camera specs; order fixes output frame order |
| `ref_cam_images` | `list[str \| bytes \| ndarray]` | Yes | One reference image per camera (path, bytes, or RGB array) |
| `ref_qpos` | `list[float] \| ndarray` | Yes | Complete MuJoCo qpos vector for the reference frame |
| `robot2world_transform` | `ndarray \| None` | No | 4x4 robot-to-world matrix; omit for fixed-base |
| `end_effectors` | `list[EESpec] \| None` | No | End-effector specs (aligned with OSS) |
| `skeleton` | `SkeletonSpec \| None` | No | Skeleton topology spec (aligned with OSS) |
| `target_size` | `tuple[int, int]` | No | Generation resolution (height, width); default (384, 640) |
| `seed` | `int` | No | RNG seed; default 1 |
| `session_id` | `str \| None` | No | Custom session ID; auto-generated if omitted |
| `timeout` | `float` | No | HTTP timeout in seconds; default 600 |

### `session.step(qpos, *, robot2world_transforms=None)`

Generate video frames for a chunk of motion.

**Parameters:**

| Parameter | Type | Required | Description |
|---|---|---|---|
| `qpos` | `ndarray` or `list` | Yes | Motion frames `(N, nq)`. Server rounds N up to a multiple of 4 |
| `robot2world_transforms` | `list[ndarray \| None] \| None` | No | Per-frame 4x4 transforms; None = identity |

**Returns:** `dict[str, list[np.ndarray]]` -- `{camera_name: [RGB uint8 (H,W,3), ...]}`

### `session.finish()`

Release the session. Idempotent. Called automatically by `__exit__`.

## Spec Classes

The SDK reuses Spec classes from `uranus.skeleton` when the full Uranus
package is installed.  Lightweight fallbacks are provided when Uranus is
not available -- they serialize to the same JSON.

### `CameraSpec(name: str)`

### `EESpec(object_type, object_name, radius_mode, pad_bodies?, radius?, sh_correction?)`

### `SkeletonSpec(mode, chains, skip_bodies, gripper_keypoint_overrides?)`

## Error Handling

```python
from uranus_sdk import UranusSDKError

try:
    session = UranusSimulationSession(...)
except UranusSDKError as e:
    print(f"Failed: {e} (code={e.code}, status={e.status_code})")
```

| Scenario | HTTP Status | Code |
|---|---|---|
| Invalid parameters | 400 | `invalid_payload` |
| Session not found | 404 | `unknown_session` |
| Session already exists | 409 | `session_exists` |
| Server busy | 503 | `server_busy` |
| Auth failure | 401 | -- |

## License

MIT
