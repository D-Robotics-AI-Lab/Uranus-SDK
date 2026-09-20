# Uranus SDK

Python SDK for the Uranus video-generation serving API.  Provides a
high-level `UranusSimulationSession` class that wraps HTTP communication,
multipart parsing, session management, and authentication.

## Installation

```bash
uv add uranus-sdk
```

Or with pip:

```bash
pip install uranus-sdk
```

## Quick Start

```python
import os
import numpy as np
from uranus.skeleton import CameraSpec, EESpec, SkeletonSpec
from uranus_sdk import UranusSimulationSession

os.environ["URANUS_BASE_URL"] = "http://localhost:8000"
os.environ["URANUS_API_KEY"] = "sk-your-token"  # optional

with UranusSimulationSession(
    prompt="pick up the cup",
    mjcf_path="mjcf/g1_no_mesh.xml",
    cameras=[
        CameraSpec(name="head"),
        CameraSpec(name="hand_left"),
        CameraSpec(name="hand_right"),
    ],
    ref_cam_images=[
        "ref_images/head.png",
        "ref_images/hand_left.png",
        "ref_images/hand_right.png",
    ],
    ref_qpos=np.load("ref_qpos.npy"),
    end_effectors=[
        EESpec(
            object_type="site",
            object_name="gripper_center",
            radius_mode="pad_pair",
            pad_bodies=("left_Left_Pad_Link", "left_Right_Pad_Link"),
        ),
    ],
    skeleton=SkeletonSpec(
        mode="chains",
        chains=(("Link1_l", "Link2_l", "Link3_l"),),
    ),
) as session:
    action = np.load("action.npy")  # (T, nq)
    for start in range(0, len(action), 4):
        frames = session.step(qpos=action[start:start + 4])
        # frames = {"head": [(384, 640, 3) RGB uint8, ...], ...}
```

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
