"""UranusSimulationSession — high-level SDK for the Uranus serving API.

Construction triggers ``create``; ``step`` returns RGB frames; ``finish``
releases the session.  Server URL and auth are read from environment
variables ``URANUS_BASE_URL`` and ``URANUS_API_KEY``.

The Spec classes (``CameraSpec``, ``EESpec``, ``SkeletonSpec``) are imported
from ``uranus.skeleton`` when the full Uranus package is installed.  This
module also provides lightweight fallback dataclasses so the SDK can be used
standalone (without the heavy ``uranus`` model package) — the fallbacks are
structurally identical and serialize to the same JSON.
"""

from __future__ import annotations

import base64
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
import requests

# ---------------------------------------------------------------------------
# Spec classes — aligned with uranus.skeleton.specs
# ---------------------------------------------------------------------------

try:
    # Use the real Spec classes from the Uranus package when available.
    from uranus.skeleton import CameraSpec, EESpec, GripperKeypointOverride, SkeletonSpec
except ImportError:  # pragma: no cover — fallback when uranus is not installed
    @dataclass(frozen=True)
    class CameraSpec:
        """One XML-defined camera of the rig."""
        name: str

        def validate(self) -> None:
            assert bool(self.name), "camera name must be non-empty"

    @dataclass(frozen=True)
    class EESpec:
        """One end-effector SH sphere: anchor object + radius source."""
        object_type: str
        object_name: str
        radius_mode: str
        pad_bodies: tuple[str, str] | None = None
        radius: float | None = None
        sh_correction: np.ndarray | None = None

    @dataclass(frozen=True)
    class GripperKeypointOverride:
        ee_object_type: str
        ee_object_name: str
        finger_bodies: tuple[str, str]
        closing_axis: int = 1

    @dataclass(frozen=True)
    class SkeletonSpec:
        mode: str = "full_tree"
        chains: tuple[tuple[str, ...], ...] = ()
        skip_bodies: tuple[str, ...] = ()
        gripper_keypoint_overrides: tuple[GripperKeypointOverride, ...] = ()


# ---------------------------------------------------------------------------
# Error
# ---------------------------------------------------------------------------

class UranusSDKError(RuntimeError):
    """Raised when the server returns an error or the network fails."""

    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        status_code: int | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.status_code = status_code


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _spec_to_dict(obj: Any) -> Any:
    """Recursively convert a Spec dataclass (or nested structure) to a JSON-
    safe dict, converting numpy arrays to nested lists."""
    if hasattr(obj, "__dataclass_fields__"):
        return {k: _spec_to_dict(v) for k, v in asdict(obj).items()}  # type: ignore[arg-type]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (list, tuple)):
        return [_spec_to_dict(v) for v in obj]
    return obj


def _load_image(img: str | Path | bytes | np.ndarray) -> bytes:
    """Accept a file path, raw bytes, or RGB ndarray and return PNG bytes."""
    if isinstance(img, (str, Path)):
        return Path(img).read_bytes()
    if isinstance(img, np.ndarray):
        return cv2.imencode(".png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))[1].tobytes()
    return img  # assume raw bytes


# ---------------------------------------------------------------------------
# Multipart decode
# ---------------------------------------------------------------------------

def _decode_multipart(
    resp: requests.Response,
) -> tuple[dict[str, Any], dict[str, list[np.ndarray]]]:
    """Parse a ``multipart/mixed`` step response.

    Returns ``(info, frames)`` where *info* is the JSON metadata dict and
    *frames* maps ``{camera_name: [BGR ndarray, ...]}``.
    """
    raw = resp.content
    boundary = raw[: raw.index(b"\r\n")]
    sections = raw.split(b"\r\n" + boundary)[:-1]

    parts: list[tuple[dict[bytes, bytes], bytes]] = []
    for section in sections:
        header_bytes, _, payload = section.partition(b"\r\n\r\n")
        headers = {}
        for line in header_bytes.split(b"\r\n"):
            if b": " in line:
                k, v = line.split(b": ", 1)
                headers[k] = v
        parts.append((headers, payload))

    info = json.loads(parts[0][1])
    frames: dict[str, list[np.ndarray]] = {}
    for headers, jpeg in parts[1:]:
        cam = headers[b"X-Camera"].decode()
        img = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
        frames.setdefault(cam, []).append(img)
    return info, frames


# ---------------------------------------------------------------------------
# UranusSimulationSession
# ---------------------------------------------------------------------------

class UranusSimulationSession:
    """High-level SDK for the Uranus video-generation serving API.

    Construction triggers ``POST /create`` — the session is ready to use as
    soon as ``__init__`` returns.  Call :meth:`step` to generate video frames
    and :meth:`finish` (or use ``with``) to release the session.

    Environment variables:
        ``URANUS_BASE_URL`` — server address (default ``http://localhost:8000``)
        ``URANUS_API_KEY``  — Bearer token; when set, every request carries
        ``Authorization: Bearer <token>``.
    """

    def __init__(
        self,
        *,
        prompt: str,
        mjcf_path: str | Path,
        cameras: Sequence[CameraSpec],
        ref_cam_images: Sequence[str | Path | bytes | np.ndarray],
        ref_qpos: list[float] | np.ndarray,
        robot2world_transform: np.ndarray | None = None,
        end_effectors: Sequence[EESpec] | None = None,
        skeleton: SkeletonSpec | None = None,
        target_size: tuple[int, int] = (384, 640),
        seed: int = 1,
        session_id: str | None = None,
        timeout: float = 600.0,
    ) -> None:
        """Create a new generation session.

        Args:
            prompt:  Natural-language task description.
            mjcf_path:  Path to the MJCF XML file (read and sent inline).
            cameras:  Ordered :class:`CameraSpec` list; order fixes output
                frame ordering for the whole session.
            ref_cam_images:  One reference image per camera.  Each entry may
                be a file path, raw bytes, or an RGB ``ndarray``.
            ref_qpos:  Complete MuJoCo ``qpos`` vector for the reference
                frame (length = ``model.nq``).
            robot2world_transform:  Optional 4x4 robot-to-world matrix.
                Omit for fixed-base robots (defaults to identity).
            end_effectors:  :class:`EESpec` list (aligned with OSS).
            skeleton:  :class:`SkeletonSpec` (aligned with OSS).
            target_size:  Generation resolution ``(height, width)``.
            seed:  RNG seed for reproducible generation.
            session_id:  Optional custom session ID; auto-generated if
                omitted.
            timeout:  Per-request HTTP timeout in seconds.

        Raises:
            UranusSDKError: If the create request fails.
        """
        # --- server config from env ---
        base_url = os.environ.get("URANUS_BASE_URL", "http://localhost:8000")
        api_key = os.environ.get("URANUS_API_KEY")
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._headers: dict[str, str] = {"Content-Type": "application/json"}
        if api_key:
            self._headers["Authorization"] = f"Bearer {api_key}"
        self._session_id: str | None = None

        # --- read MJCF ---
        mjcf_xml = Path(mjcf_path).read_text(encoding="utf-8")
        camera_names = [c.name for c in cameras]

        # --- ref_qpos -> frame dict ---
        ref_frame: dict[str, Any] = {
            "state": np.asarray(ref_qpos, dtype=np.float64).tolist(),
        }
        if robot2world_transform is not None:
            ref_frame["robot2world_transform"] = np.asarray(
                robot2world_transform, dtype=np.float64
            ).tolist()

        # --- ref_cam_images -> bytes ---
        ref_images_bytes = [_load_image(img) for img in ref_cam_images]

        # --- Spec -> dict ---
        ee_dicts = [_spec_to_dict(ee) for ee in (end_effectors or [])]
        skel_dict = _spec_to_dict(skeleton) if skeleton is not None else {}

        # --- payload ---
        payload = {
            "camera_names": camera_names,
            "prompt": prompt,
            "seed": seed,
            "target_size": list(target_size),
            "mjcf_xml": mjcf_xml,
            "end_effectors": ee_dicts,
            "skeleton": skel_dict,
            "ref_qpos": ref_frame,
        }
        body: dict[str, Any] = {
            "payload": payload,
            "attachments": {
                "ref_cam_images": [
                    base64.b64encode(img).decode() for img in ref_images_bytes
                ]
            },
        }
        if session_id is not None:
            body["session_id"] = session_id

        # --- POST /create ---
        resp = requests.post(
            f"{self._base_url}/create",
            json=body,
            headers=self._headers,
            timeout=self._timeout,
        )
        self._raise_for_error(resp)
        self._session_id = resp.json()["session_id"]

    # ------------------------------------------------------------------
    # step
    # ------------------------------------------------------------------

    def step(
        self,
        qpos: list[list[float]] | np.ndarray,
        *,
        robot2world_transforms: list[np.ndarray | None] | None = None,
    ) -> dict[str, list[np.ndarray]]:
        """Generate video frames for a chunk of motion.

        Args:
            qpos:  Motion frame sequence ``(N, nq)``.  Each row is a complete
                MuJoCo ``qpos`` vector.  The server rounds ``N`` up to a
                multiple of ``temporal_interval`` (=4).
            robot2world_transforms:  Optional per-frame 4x4 transforms.
                ``None`` or a frame entry ``None`` → identity matrix.

        Returns:
            ``{camera_name: [RGB uint8 ndarray (H, W, 3), ...]}``

        Raises:
            UranusSDKError: On server error or if the session was finished.
        """
        if self._session_id is None:
            raise UranusSDKError("Session not created or already finished")

        qpos_arr = np.asarray(qpos, dtype=np.float64)
        if qpos_arr.ndim == 1:
            qpos_arr = qpos_arr[np.newaxis, :]

        frames: list[dict[str, Any]] = []
        for i in range(len(qpos_arr)):
            frame: dict[str, Any] = {"state": qpos_arr[i].tolist()}
            if robot2world_transforms is not None:
                t = robot2world_transforms[i]
                if t is not None:
                    frame["robot2world_transform"] = np.asarray(
                        t, dtype=np.float64
                    ).tolist()
            frames.append(frame)

        payload = {"qpos": frames, "num_step": len(qpos_arr)}

        resp = requests.post(
            f"{self._base_url}/step",
            json={"session_id": self._session_id, "payload": payload},
            headers=self._headers,
            timeout=self._timeout,
        )
        self._raise_for_error(resp)
        _info, bgr_frames = _decode_multipart(resp)

        return {
            cam: [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in imgs]
            for cam, imgs in bgr_frames.items()
        }

    # ------------------------------------------------------------------
    # finish
    # ------------------------------------------------------------------

    def finish(self) -> None:
        """Release the session (GPU/CPU state).  Idempotent."""
        if self._session_id is not None:
            try:
                requests.delete(
                    f"{self._base_url}/sessions/{self._session_id}",
                    headers=self._headers,
                    timeout=self._timeout,
                )
            except Exception:
                pass
            self._session_id = None

    # ------------------------------------------------------------------
    # context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> UranusSimulationSession:
        return self

    def __exit__(self, *args: object) -> None:
        self.finish()

    # ------------------------------------------------------------------
    # error handling
    # ------------------------------------------------------------------

    @staticmethod
    def _raise_for_error(resp: requests.Response) -> None:
        if resp.status_code >= 400:
            try:
                data = resp.json()
                err = data.get("error", data)
                msg = err.get("message", resp.text)
                code = err.get("code")
            except Exception:
                msg, code = resp.text, None
            raise UranusSDKError(
                f"HTTP {resp.status_code}: {msg}",
                code=code,
                status_code=resp.status_code,
            )
