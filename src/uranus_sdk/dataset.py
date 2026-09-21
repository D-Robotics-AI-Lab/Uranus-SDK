"""Load sample data from HuggingFace Hub via direct HTTP (no huggingface_hub).

Usage:
    sample = load_sample("D-Robotics/Uranus-Demo-Data/resolve/main/rc-table30-v1__rc-aloha/000000")
    # sample["create"] → kwargs for UranusSimulationSession
    # sample["steps"]  → list of {qpos, robot2world_transforms} dicts
"""

from __future__ import annotations

import json
import os
from typing import Any

import numpy as np
import requests

from uranus_sdk.specs import (
    CameraSpec,
    EESpec,
    GripperKeypointOverride,
    SkeletonSpec,
)

_HF_BASE = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/") + "/datasets"


def _hf_url(hf_path: str, filename: str) -> str:
    """Build a HuggingFace direct-download URL.

    ``hf_path`` is everything after ``huggingface.co/datasets/``, e.g.
    ``"D-Robotics/Uranus-Demo-Data/resolve/main/rc-table30-v1__rc-aloha/000000"``.
    """
    return f"{_HF_BASE}/{hf_path}/{filename}"


def _download_text(hf_path: str, filename: str) -> str:
    """Download a text file from HF Hub."""
    resp = requests.get(_hf_url(hf_path, filename), timeout=60)
    resp.raise_for_status()
    return resp.text


def _download_bytes(hf_path: str, filename: str) -> bytes:
    """Download a binary file from HF Hub."""
    resp = requests.get(_hf_url(hf_path, filename), timeout=120)
    resp.raise_for_status()
    return resp.content


def load_sample(
    hf_path: str,
    *,
    num_chunks: int | None = None,
    step_length: int = 4,
) -> dict[str, Any]:
    """Load a sample from HuggingFace Hub and build SDK-ready data.

    Args:
        hf_path:    The path after ``huggingface.co/datasets/``, e.g.
                    ``"D-Robotics/Uranus-Demo-Data/resolve/main/rc-table30-v1__rc-aloha/000000"``.
        num_chunks: Number of step chunks to generate. ``None`` → all available.
        step_length: Frames per step (default 4, = temporal_interval).

    Returns:
        A dict with two keys:

        ``"create"``:  kwargs dict for ``UranusSimulationSession(**sample["create"])``
        ``"steps"``:  list of ``{"qpos": ndarray, "robot2world_transforms": list}`` dicts

    Example::

        sample = load_sample("D-Robotics/Uranus-Demo-Data/resolve/main/rc-table30-v1__rc-aloha/000000")
        with UranusSimulationSession(**sample["create"]) as session:
            for step in sample["steps"]:
                frames = session.step(**step)
    """
    # --- download meta + temporal ---
    meta = json.loads(_download_text(hf_path, "meta.json"))
    temporal = json.loads(_download_text(hf_path, "temporal.json"))

    camera_names = list(meta["cameras"])
    cameras = [CameraSpec(name=name) for name in camera_names]

    # --- download MJCF ---
    mjcf_filename = meta["mjcf_path"]
    # mjcf_path may be relative ("mjcf/robot.xml") — strip any dir prefix
    if "/" in mjcf_filename:
        mjcf_filename = mjcf_filename.split("/")[-1]
    # try with the original path first, then fallback to basename
    try:
        mjcf_xml = _download_text(hf_path, meta["mjcf_path"])
    except requests.HTTPError:
        mjcf_xml = _download_text(hf_path, f"mjcf/{mjcf_filename}")

    # --- download reference images ---
    ref_cam_images = []
    for name in camera_names:
        img_path = meta["ref_cam_images"][name]
        try:
            ref_cam_images.append(_download_bytes(hf_path, img_path))
        except requests.HTTPError:
            # fallback: strip directory prefix
            basename = img_path.split("/")[-1]
            ref_cam_images.append(_download_bytes(hf_path, f"ref_images/{basename}"))

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
        "target_size": (int(meta.get("height", 384)), int(meta.get("width", 640))),
        "seed": 1,
    }

    # --- build steps ---
    available = len(states) - 1  # frame 0 is reference
    if num_chunks is None:
        num_chunks = available // step_length

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
