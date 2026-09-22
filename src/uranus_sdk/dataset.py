"""Load sample data from a local directory or an HTTP URL.

Usage:
    # From HuggingFace Hub (HTTP):
    sample = load_sample("https://huggingface.co/datasets/D-Robotics/Uranus-Demo-Data/resolve/main/rc-table30-v1__rc-aloha/000000")

    # From a mirror:
    sample = load_sample("https://hf-mirror.com/datasets/D-Robotics/Uranus-Demo-Data/resolve/main/rc-table30-v1__rc-aloha/000000")

    # From a local directory:
    sample = load_sample("/path/to/sample/000000")

    # sample["create"] → kwargs for UranusSimulationSession
    # sample["steps"]  → list of {qpos, robot2world_transforms} dicts
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from urllib.parse import quote

import numpy as np
import requests

from uranus_sdk.specs import (
    CameraSpec,
    EESpec,
    GripperKeypointOverride,
    SkeletonSpec,
)


# ---------------------------------------------------------------------------
# I/O helpers — local or HTTP, auto-detected by the root path prefix
# ---------------------------------------------------------------------------

class _LocalReader:
    """Read files from a local directory."""

    def __init__(self, root: str | Path) -> None:
        self._root = Path(root)

    def read_text(self, filename: str) -> str:
        return (self._root / filename).read_text(encoding="utf-8")

    def read_bytes(self, filename: str) -> bytes:
        return (self._root / filename).read_bytes()


class _HttpReader:
    """Download files from an HTTP base URL (e.g. HuggingFace Hub)."""

    def __init__(self, base_url: str) -> None:
        # Honor HF_ENDPOINT: if the URL points to huggingface.co and the env
        # var is set, swap the host so the request goes through the mirror.
        hf_endpoint = os.environ.get("HF_ENDPOINT")
        if hf_endpoint and "huggingface.co" in base_url:
            base_url = base_url.replace("https://huggingface.co", hf_endpoint.rstrip("/"))
            base_url = base_url.replace("http://huggingface.co", hf_endpoint.rstrip("/"))
        self._base = base_url.rstrip("/")

    def _url(self, filename: str) -> str:
        """Join the base URL with a filename, preserving path slashes."""
        # urljoin needs a trailing slash on the base for relative paths to work
        return f"{self._base}/{quote(filename, safe='/')}"

    def read_text(self, filename: str) -> str:
        resp = requests.get(self._url(filename), timeout=60)
        resp.raise_for_status()
        return resp.text

    def read_bytes(self, filename: str) -> bytes:
        resp = requests.get(self._url(filename), timeout=300)
        resp.raise_for_status()
        return resp.content


def _make_reader(root: str) -> _LocalReader | _HttpReader:
    """Create a reader based on whether *root* is an HTTP URL or a local path."""
    if root.startswith(("http://", "https://")):
        return _HttpReader(root)
    return _LocalReader(root)


# ---------------------------------------------------------------------------
# load_sample
# ---------------------------------------------------------------------------

def load_sample(
    root: str,
    *,
    num_chunks: int | None = None,
    step_length: int = 4,
) -> dict[str, Any]:
    """Load a sample from a local directory or an HTTP URL and build SDK-ready data.

    Args:
        root:       Sample root path. Either:

                    * An HTTP URL, e.g.
                      ``"https://huggingface.co/datasets/D-Robotics/Uranus-Demo-Data/resolve/main/rc-table30-v1__rc-aloha/000000"``
                    * A local directory, e.g.
                      ``"/path/to/sample/000000"``

                    The root must contain ``meta.json``, ``temporal.json``,
                    the MJCF file, and reference images.
        num_chunks: Number of step chunks to generate. ``None`` → all available.
        step_length: Frames per step (default 4, = temporal_interval).

    Returns:
        A dict with two keys:

        ``"create"``:  kwargs dict for ``UranusSimulationSession(**sample["create"])``
        ``"steps"``:  list of ``{"qpos": ndarray, "robot2world_transforms": list}`` dicts

    Example::

        # From HuggingFace
        sample = load_sample("https://huggingface.co/datasets/D-Robotics/Uranus-Demo-Data/resolve/main/rc-table30-v1__rc-aloha/000000")

        # From local disk
        sample = load_sample("/data/samples/000000")

        with UranusSimulationSession(**sample["create"]) as session:
            for step in sample["steps"]:
                frames = session.step(**step)
    """
    reader = _make_reader(root)

    # --- read meta + temporal ---
    meta = json.loads(reader.read_text("meta.json"))
    temporal = json.loads(reader.read_text("temporal.json"))

    camera_names = list(meta["cameras"])
    cameras = [CameraSpec(name=name) for name in camera_names]

    # --- read MJCF ---
    mjcf_path = meta["mjcf_path"]
    mjcf_xml = reader.read_text(mjcf_path)

    # --- read reference images ---
    ref_cam_images = [reader.read_bytes(meta["ref_cam_images"][name]) for name in camera_names]

    # --- build end_effectors ---
    end_effectors = []
    for ee in meta.get("end_effectors", []):
        end_effectors.append(EESpec(
            object_type=ee["object_type"],
            object_name=ee["object_name"],
            radius_mode=ee["radius_mode"],
            pad_bodies=tuple(ee.get("pad_bodies", [])) if ee.get("pad_bodies") else None,
            radius=float(ee["radius"]) if ee.get("radius") is not None else None,
            sh_correction=np.array(ee["sh_correction"]) if ee.get("sh_correction") else None,
        ))

    # --- build skeleton ---
    skel = meta.get("skeleton", {})
    overrides = tuple(
        GripperKeypointOverride(
            ee_object_type=str(ov["ee_object_type"]),
            ee_object_name=str(ov["ee_object_name"]),
            finger_bodies=tuple(ov["finger_bodies"]),
            closing_axis=int(ov.get("closing_axis", 1)),
        )
        for ov in skel.get("gripper_keypoint_overrides", [])
    )
    skeleton = SkeletonSpec(
        mode=skel.get("mode", "full_tree"),
        chains=tuple(tuple(c) for c in skel.get("chains", [])),
        skip_bodies=tuple(skel.get("skip_bodies", [])),
        gripper_keypoint_overrides=overrides,
    )

    # --- reference frame ---
    states = temporal["step_qpos"]
    ref_frame = states[0]
    ref_qpos = ref_frame["state"]
    r2w = (
        np.array(ref_frame["robot2world_transform"])
        if "robot2world_transform" in ref_frame
        else None
    )

    # --- build create kwargs ---
    create_kwargs = {
        "prompt": meta.get("prompt", ""),
        "mjcf_xml": mjcf_xml,
        "cameras": cameras,
        "ref_cam_images": ref_cam_images,
        "ref_qpos": ref_qpos,
        "robot2world_transform": r2w,
        "end_effectors": end_effectors,
        "skeleton": skeleton,
        "target_size": (384, 640),
        "seed": 1,
    }

    # --- build steps ---
    available = len(states) - 1  # frame 0 is reference
    max_chunks = available // step_length
    if num_chunks is None or num_chunks > max_chunks:
        num_chunks = max_chunks

    steps = []
    for i in range(num_chunks):
        start = i * step_length + 1
        chunk = states[start : start + step_length]
        if len(chunk) < step_length:
            chunk = chunk + [chunk[-1]] * (step_length - len(chunk))

        qpos = np.array([f["state"] for f in chunk])
        r2w_list = [
            np.array(f["robot2world_transform"])
            if "robot2world_transform" in f
            else None
            for f in chunk
        ]
        steps.append({"qpos": qpos, "robot2world_transforms": r2w_list})

    return {"create": create_kwargs, "steps": steps}
