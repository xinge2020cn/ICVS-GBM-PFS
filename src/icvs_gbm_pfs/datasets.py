"""Patient-level volumetric datasets and joint four-channel augmentation."""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as functional
from torch.utils.data import Dataset

from .config import StudyConfig
from .preprocessing import load_cropped_volume


class JointMRITransform:
    """Apply one spatial transform to all MRI channels and channelwise intensity changes."""

    def __init__(self, *, seed: int) -> None:
        self.generator = torch.Generator().manual_seed(seed)

    def _uniform(self, lower: float, upper: float) -> float:
        value = torch.rand((), generator=self.generator).item()
        return lower + (upper - lower) * value

    @staticmethod
    def _gaussian_smooth(image: torch.Tensor, sigma: float) -> torch.Tensor:
        if not math.isfinite(sigma) or sigma <= 0:
            raise ValueError("Gaussian smoothing sigma must be finite and greater than zero.")
        radius = max(1, int(math.ceil(3.0 * sigma)))
        coordinates = torch.arange(
            -radius,
            radius + 1,
            dtype=image.dtype,
            device=image.device,
        )
        kernel_1d = torch.exp(-0.5 * (coordinates / sigma).square())
        kernel_1d = kernel_1d / kernel_1d.sum()
        kernel_3d = (
            kernel_1d[:, None, None]
            * kernel_1d[None, :, None]
            * kernel_1d[None, None, :]
        )
        channels = image.shape[0]
        weight = kernel_3d.expand(channels, 1, -1, -1, -1).contiguous()
        return functional.conv3d(
            image.unsqueeze(0),
            weight,
            padding=radius,
            groups=channels,
        ).squeeze(0)

    def __call__(self, image: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        channels, depth, height, width = image.shape
        if mask is None:
            mask = image.abs().sum(dim=0, keepdim=True).gt(0).to(image.dtype)
        if mask.shape != (1, depth, height, width):
            raise ValueError("The augmentation mask must have shape (1, depth, height, width).")
        mask = mask.to(dtype=image.dtype, device=image.device)
        angle_x = math.radians(self._uniform(-10.0, 10.0))
        angle_y = math.radians(self._uniform(-10.0, 10.0))
        angle_z = math.radians(self._uniform(-10.0, 10.0))
        scale = self._uniform(0.90, 1.10)
        translation_x = self._uniform(-5.0, 5.0) * 2.0 / max(width - 1, 1)
        translation_y = self._uniform(-5.0, 5.0) * 2.0 / max(height - 1, 1)
        translation_z = self._uniform(-1.0, 1.0) * 2.0 / max(depth - 1, 1)
        cosine_x, sine_x = math.cos(angle_x), math.sin(angle_x)
        cosine_y, sine_y = math.cos(angle_y), math.sin(angle_y)
        cosine_z, sine_z = math.cos(angle_z), math.sin(angle_z)
        rotation_x = image.new_tensor(
            [[1.0, 0.0, 0.0], [0.0, cosine_x, -sine_x], [0.0, sine_x, cosine_x]]
        )
        rotation_y = image.new_tensor(
            [[cosine_y, 0.0, sine_y], [0.0, 1.0, 0.0], [-sine_y, 0.0, cosine_y]]
        )
        rotation_z = image.new_tensor(
            [[cosine_z, -sine_z, 0.0], [sine_z, cosine_z, 0.0], [0.0, 0.0, 1.0]]
        )
        linear = (rotation_z @ rotation_y @ rotation_x) / scale
        translation = image.new_tensor([translation_x, translation_y, translation_z])
        theta = torch.cat([linear, translation[:, None]], dim=1).unsqueeze(0)
        expanded = image.unsqueeze(0)
        grid = functional.affine_grid(theta, expanded.shape, align_corners=False)
        image = functional.grid_sample(
            expanded,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        ).squeeze(0)
        mask = functional.grid_sample(
            mask.unsqueeze(0),
            grid,
            mode="nearest",
            padding_mode="zeros",
            align_corners=False,
        ).squeeze(0)
        intensity_scale = self._uniform(0.90, 1.10)
        intensity_shift = self._uniform(-0.10, 0.10)
        image = image * intensity_scale + intensity_shift
        noise_standard_deviation = self._uniform(0.0, 0.05)
        noise = torch.randn(
            image.shape,
            dtype=image.dtype,
            device=image.device,
            generator=self.generator,
        )
        image = image + noise * noise_standard_deviation
        if self._uniform(0.0, 1.0) < 0.30:
            image = self._gaussian_smooth(image, self._uniform(0.50, 1.00))
        image = image * mask
        return image.reshape(channels, depth, height, width).contiguous()


class SurvivalVolumeDataset(Dataset[dict[str, object]]):
    """Load one four-channel tumor-peritumoral volume per patient."""

    def __init__(
        self,
        frame: pd.DataFrame,
        config: StudyConfig,
        *,
        augment: bool,
        cache_dir: str | Path | None = None,
        seed_offset: int = 0,
    ) -> None:
        self.frame = frame.reset_index(drop=True).copy()
        self.config = config
        shape = config.section("preprocessing")["target_shape_dhw"]
        self.target_shape = tuple(int(value) for value in shape)
        self.transform = JointMRITransform(seed=config.seed + seed_offset) if augment else None
        self.cache_dir = Path(cache_dir).resolve() if cache_dir is not None else None
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def __len__(self) -> int:
        return len(self.frame)

    def _cache_path(self, index: int) -> Path | None:
        if self.cache_dir is None:
            return None
        row = self.frame.iloc[index]
        patient_id = str(row[self.config.column("patient_id")])
        safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", patient_id)
        sources = []
        for column in (
            "preprocessed_t1_path",
            "preprocessed_t2_path",
            "preprocessed_flair_path",
            "preprocessed_ce_t1_path",
            "voi_path",
        ):
            path = Path(str(row[column])).resolve()
            stat = path.stat()
            sources.append(
                {
                    "path": str(path),
                    "size": stat.st_size,
                    "modified_ns": stat.st_mtime_ns,
                }
            )
        payload = json.dumps(
            {"patient_id": patient_id, "target_shape_dhw": self.target_shape, "sources": sources},
            sort_keys=True,
        ).encode("utf-8")
        digest = hashlib.sha256(payload).hexdigest()[:16]
        return self.cache_dir / f"{safe_id}-{digest}.pt"

    def _load_image(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        cache_path = self._cache_path(index)
        if cache_path is not None and cache_path.is_file():
            cached = torch.load(cache_path, map_location="cpu", weights_only=True)
            if not isinstance(cached, dict) or set(cached) != {"image", "mask"}:
                raise ValueError(f"Invalid volume cache entry: {cache_path}")
            image = cached["image"]
            mask = cached["mask"]
            expected_image_shape = (4, *self.target_shape)
            expected_mask_shape = (1, *self.target_shape)
            if image.shape != expected_image_shape or mask.shape != expected_mask_shape:
                raise ValueError(
                    f"Cached volume geometry does not match the configuration: {cache_path}"
                )
            if not torch.isfinite(image).all() or not torch.isfinite(mask).all():
                raise ValueError(f"Cached volume contains nonfinite values: {cache_path}")
            return image, mask
        image, mask = load_cropped_volume(self.frame.iloc[index], self.target_shape)
        if cache_path is not None:
            temporary = cache_path.with_suffix(".partial")
            torch.save({"image": image, "mask": mask}, temporary)
            temporary.replace(cache_path)
        return image, mask

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.frame.iloc[index]
        image, mask = self._load_image(index)
        if self.transform is not None:
            image = self.transform(image, mask)
        return {
            "image": image,
            "time": torch.tensor(float(row[self.config.column("pfs_time")]), dtype=torch.float32),
            "event": torch.tensor(int(row[self.config.column("pfs_event")]), dtype=torch.bool),
            "patient_id": str(row[self.config.column("patient_id")]),
        }
