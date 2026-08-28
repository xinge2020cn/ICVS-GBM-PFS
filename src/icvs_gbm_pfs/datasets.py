"""Patient-level volumetric datasets and joint four-channel augmentation."""

from __future__ import annotations

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

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        channels, depth, height, width = image.shape
        angle = math.radians(self._uniform(-10.0, 10.0))
        scale = self._uniform(0.90, 1.10)
        translation_x = self._uniform(-5.0, 5.0) * 2.0 / max(width - 1, 1)
        translation_y = self._uniform(-5.0, 5.0) * 2.0 / max(height - 1, 1)
        translation_z = self._uniform(-1.0, 1.0) * 2.0 / max(depth - 1, 1)
        cosine = math.cos(angle) / scale
        sine = math.sin(angle) / scale
        theta = image.new_tensor(
            [
                [cosine, -sine, 0.0, translation_x],
                [sine, cosine, 0.0, translation_y],
                [0.0, 0.0, 1.0 / scale, translation_z],
            ]
        ).unsqueeze(0)
        expanded = image.unsqueeze(0)
        grid = functional.affine_grid(theta, expanded.shape, align_corners=False)
        image = functional.grid_sample(
            expanded,
            grid,
            mode="bilinear",
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
            image = functional.avg_pool3d(
                image.unsqueeze(0), kernel_size=3, stride=1, padding=1
            ).squeeze(0)
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

    def _load_image(self, index: int) -> torch.Tensor:
        patient_id = str(self.frame.iloc[index][self.config.column("patient_id")])
        safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", patient_id)
        cache_path = self.cache_dir / f"{safe_id}.pt" if self.cache_dir is not None else None
        if cache_path is not None and cache_path.is_file():
            return torch.load(cache_path, map_location="cpu", weights_only=True)
        image, _ = load_cropped_volume(self.frame.iloc[index], self.target_shape)
        if cache_path is not None:
            temporary = cache_path.with_suffix(".partial")
            torch.save(image, temporary)
            temporary.replace(cache_path)
        return image

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.frame.iloc[index]
        image = self._load_image(index)
        if self.transform is not None:
            image = self.transform(image)
        return {
            "image": image,
            "time": torch.tensor(float(row[self.config.column("pfs_time")]), dtype=torch.float32),
            "event": torch.tensor(int(row[self.config.column("pfs_event")]), dtype=torch.bool),
            "patient_id": str(row[self.config.column("patient_id")]),
        }
