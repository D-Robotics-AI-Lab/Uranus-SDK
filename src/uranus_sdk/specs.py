"""Spec classes for rig configuration — structurally compatible with
``uranus.skeleton.specs``.

These dataclasses serialize (via :func:`dataclasses.asdict`) to the same
JSON shape that the server's ``compat.from_create_payload`` expects.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


@dataclass(frozen=True)
class CameraSpec:
    """One XML-defined camera of the rig."""

    name: str

    def validate(self) -> None:
        _require(bool(self.name), "camera name must be non-empty")


@dataclass(frozen=True)
class EESpec:
    """One end-effector SH sphere: anchor object + radius source.

    Fields mirror ``uranus.skeleton.specs.EESpec``.
    """

    object_type: str  # "body" | "site"
    object_name: str
    radius_mode: str  # "pad_pair" | "fixed"
    pad_bodies: tuple[str, str] | None = None
    radius: float | None = None  # fixed mode, meters
    sh_correction: np.ndarray | None = None  # 3x3, default identity

    def validate(self) -> None:
        _require(
            self.object_type in ("body", "site"),
            f"end_effectors[{self.object_name!r}].object_type must be 'body' or 'site'",
        )
        _require(bool(self.object_name), "end_effector object_name must be non-empty")
        if self.radius_mode == "pad_pair":
            _require(
                self.pad_bodies is not None and len(self.pad_bodies) == 2,
                f"end_effectors[{self.object_name!r}] pad_pair requires two pad bodies",
            )
        elif self.radius_mode == "fixed":
            _require(
                self.radius is not None and self.radius > 0.0,
                f"end_effectors[{self.object_name!r}] fixed radius must be > 0",
            )
        else:
            raise ValueError(
                f"end_effectors[{self.object_name!r}].radius_mode must be 'pad_pair' or 'fixed'"
            )


@dataclass(frozen=True)
class GripperKeypointOverride:
    """Legacy ``render_gripper_keypoints_from_width`` semantics, expressed via
    the rig: the two finger-body keypoints are repositioned to
    ``EE_pos +/- EE_rot[:, closing_axis] * width/2``."""

    ee_object_type: str  # "body" | "site"
    ee_object_name: str
    finger_bodies: tuple[str, str]
    closing_axis: int = 1  # column index into the EE rotation

    def validate(self) -> None:
        _require(self.ee_object_type in ("body", "site"), "gripper override ee_object_type")
        _require(len(self.finger_bodies) == 2, "gripper override needs two finger bodies")
        _require(0 <= self.closing_axis < 3, "closing_axis must be 0/1/2")


@dataclass(frozen=True)
class SkeletonSpec:
    """Keypoint topology for rendering."""

    mode: str = "full_tree"  # "chains" | "full_tree"
    chains: tuple[tuple[str, ...], ...] = ()
    skip_bodies: tuple[str, ...] = ()
    gripper_keypoint_overrides: tuple[GripperKeypointOverride, ...] = ()

    def validate(self) -> None:
        if self.mode not in ("chains", "full_tree"):
            raise ValueError(f"skeleton.mode must be 'chains' or 'full_tree', got {self.mode!r}")
        if self.mode == "chains":
            _require(len(self.chains) > 0, "skeleton chains must be non-empty in 'chains' mode")
        for ov in self.gripper_keypoint_overrides:
            ov.validate()
