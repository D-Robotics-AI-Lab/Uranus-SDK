"""UranusSimulationSession — high-level SDK for the Uranus serving API.

Construction triggers ``create``; ``step`` returns RGB frames; ``finish``
releases the session.  Server URL and auth are read from environment
variables ``URANUS_BASE_URL`` and ``URANUS_API_KEY``.

Spec classes (``CameraSpec``, ``EESpec``, ``SkeletonSpec``) are defined
in :mod:`uranus_sdk.specs` and are structurally compatible with the
``uranus.skeleton`` Spec classes used by the server.
"""

from __future__ import annotations

import base64
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
import requests

from uranus_sdk.specs import (
    CameraSpec,
    EESpec,
    SkeletonSpec,
)

#: Default Uranus serving endpoint, used when ``URANUS_BASE_URL`` is unset.
DEFAULT_BASE_URL = "https://uranus.d-robotics.cc/api/v1"


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


def _load_image(
    img: str | Path | bytes | np.ndarray, target_size: tuple[int, int]
) -> tuple[bytes, tuple[int, int]]:
    """Decode image, resize to target_size, and return (resized PNG bytes, original_size).

    The original size is returned so the caller can scale MJCF intrinsics
    from original resolution to target resolution — the server will see
    raw_sizes == target_size and skip its own intrinsics scaling.
    """
    target_h, target_w = target_size

    # Decode to BGR ndarray
    if isinstance(img, (str, Path)):
        raw = Path(img).read_bytes()
        bgr = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    elif isinstance(img, np.ndarray):
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    else:
        bgr = cv2.imdecode(np.frombuffer(img, dtype=np.uint8), cv2.IMREAD_COLOR)

    orig_h, orig_w = bgr.shape[:2]

    # Resize to target resolution
    if (orig_h, orig_w) != (target_h, target_w):
        bgr = cv2.resize(bgr, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

    return cv2.imencode(".png", bgr)[1].tobytes(), (orig_h, orig_w)


def _scale_mjcf_intrinsics(
    mjcf_xml: str, orig_sizes: dict[str, tuple[int, int]], target_size: tuple[int, int]
) -> str:
    """Scale MJCF <camera> focal/principal from original resolution to target.

    The server reads intrinsics from ``model.cam_intrinsic`` (populated by the
    MJCF ``focal``/``principal`` attributes) and scales them based on the
    decoded image dimensions (``raw_sizes``). By pre-resizing images to
    target_size, the server sees ``raw_sizes == target_size`` and skips
    scaling. We must therefore scale the intrinsics in the XML ourselves.
    """
    import xml.etree.ElementTree as ET

    target_h, target_w = target_size
    root = ET.fromstring(mjcf_xml)

    for cam_elem in root.iter("camera"):
        name = cam_elem.get("name")
        if name not in orig_sizes:
            continue
        orig_h, orig_w = orig_sizes[name]
        scale_x = target_w / orig_w
        scale_y = target_h / orig_h

        focal = cam_elem.get("focal")
        if focal:
            fx, fy = (float(x) for x in focal.split())
            cam_elem.set("focal", f"{fx * scale_x:.12g} {fy * scale_y:.12g}")

        principal = cam_elem.get("principal")
        if principal:
            cx, cy = (float(x) for x in principal.split())
            cam_elem.set("principal", f"{cx * scale_x:.12g} {cy * scale_y:.12g}")

    return ET.tostring(root, encoding="unicode")


# ---------------------------------------------------------------------------
# Response decode
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


def _decode_json(
    resp: requests.Response,
) -> tuple[dict[str, Any], dict[str, list[np.ndarray]]]:
    """Parse the JSON step response used by serving v2.

    The response shape is ``{"session_id": ..., "output": {"obs":
    {camera: [base64 PNG, ...]}}}``.  Returns the same ``(info, frames)``
    tuple as :func:`_decode_multipart`, with frames in BGR order.
    """
    info = resp.json()
    obs = info.get("output", {}).get("obs", {}) if isinstance(info, dict) else {}

    frames: dict[str, list[np.ndarray]] = {}
    for cam, encoded in obs.items():
        imgs: list[np.ndarray] = []
        for item in encoded:
            img = cv2.imdecode(
                np.frombuffer(base64.b64decode(item), dtype=np.uint8),
                cv2.IMREAD_COLOR,
            )
            if img is None:
                raise UranusSDKError(
                    f"Failed to decode frame for camera {cam!r}"
                )
            imgs.append(img)
        frames[str(cam)] = imgs
    return info, frames


def _decode_response(
    resp: requests.Response,
) -> tuple[dict[str, Any], dict[str, list[np.ndarray]]]:
    """Decode either the JSON (serving v2) or multipart (legacy) frame response."""
    content_type = resp.headers.get("Content-Type", "").lower()
    if "multipart/" in content_type:
        return _decode_multipart(resp)
    if "json" in content_type or resp.content.lstrip().startswith(b"{"):
        return _decode_json(resp)
    return _decode_multipart(resp)


# ---------------------------------------------------------------------------
# UranusSimulationSession
# ---------------------------------------------------------------------------

class UranusSimulationSession:
    """High-level SDK for the Uranus video-generation serving API.

    Construction triggers ``POST /create`` — the session is ready to use as
    soon as ``__init__`` returns.  Call :meth:`step` to generate video frames
    and :meth:`finish` (or use ``with``) to release the session.

    Environment variables:
        ``URANUS_BASE_URL`` — server address (default
        ``https://uranus.d-robotics.cc/api/v1``)
        ``URANUS_API_KEY``  — Bearer token; when set, every request carries
        ``Authorization: Bearer <token>``.
    """

    def __init__(
        self,
        *,
        prompt: str,
        cameras: Sequence[CameraSpec],
        ref_cam_images: Sequence[str | Path | bytes | np.ndarray],
        ref_qpos: list[float] | np.ndarray,
        mjcf_path: str | Path | None = None,
        mjcf_xml: str | None = None,
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
            cameras:  Ordered :class:`CameraSpec` list; order fixes output
                frame ordering for the whole session.
            ref_cam_images:  One reference image per camera.  Each entry may
                be a file path, raw bytes, or an RGB ``ndarray``.
            ref_qpos:  Complete MuJoCo ``qpos`` vector for the reference
                frame (length = ``model.nq``).
            mjcf_path:  Path to the MJCF XML file (read and sent inline).
                Mutually exclusive with *mjcf_xml*.
            mjcf_xml:  Inline MJCF XML string (use when loading from a URL
                via :func:`load_sample`).  Mutually exclusive with *mjcf_path*.
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
        base_url = os.environ.get("URANUS_BASE_URL", DEFAULT_BASE_URL)
        api_key = os.environ.get("URANUS_API_KEY")
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._headers: dict[str, str] = {"Content-Type": "application/json"}
        if api_key:
            self._headers["Authorization"] = f"Bearer {api_key}"
        self._session_id: str | None = None

        # --- read MJCF ---
        if mjcf_xml is not None:
            pass  # use inline string directly
        elif mjcf_path is not None:
            mjcf_xml = Path(mjcf_path).read_text(encoding="utf-8")
        else:
            raise ValueError("Either mjcf_path or mjcf_xml must be provided")
        camera_names = [c.name for c in cameras]

        # --- ref_qpos -> frame dict ---
        ref_frame: dict[str, Any] = {
            "state": np.asarray(ref_qpos, dtype=np.float64).tolist(),
        }
        if robot2world_transform is not None:
            ref_frame["robot2world_transform"] = np.asarray(
                robot2world_transform, dtype=np.float64
            ).tolist()

        # --- ref_cam_images -> resized bytes + track original sizes ---
        resized_images: list[bytes] = []
        orig_sizes: dict[str, tuple[int, int]] = {}
        for cam, img in zip(cameras, ref_cam_images):
            img_bytes, orig_size = _load_image(img, target_size)
            resized_images.append(img_bytes)
            orig_sizes[cam.name] = orig_size

        # --- scale MJCF intrinsics from original to target resolution ---
        mjcf_xml = _scale_mjcf_intrinsics(mjcf_xml, orig_sizes, target_size)

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
                    base64.b64encode(img).decode() for img in resized_images
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
        _info, bgr_frames = _decode_response(resp)

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
