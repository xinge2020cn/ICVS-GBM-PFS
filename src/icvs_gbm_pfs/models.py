"""Three-dimensional neural survival architectures."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


class VisionTransformer3DSurvival(nn.Module):
    """Four-channel 3D vision transformer with a continuous Cox log-risk head."""

    def __init__(
        self,
        *,
        input_shape_dhw: Sequence[int] = (24, 192, 192),
        input_channels: int = 4,
        patch_shape_dhw: Sequence[int] = (4, 16, 16),
        embedding_dim: int = 256,
        depth: int = 6,
        heads: int = 8,
        mlp_ratio: float = 4.0,
        attention_dropout: float = 0.10,
        projection_dropout: float = 0.10,
        head_dim: int = 64,
        head_dropout: float = 0.30,
    ) -> None:
        super().__init__()
        input_shape = tuple(int(value) for value in input_shape_dhw)
        patch_shape = tuple(int(value) for value in patch_shape_dhw)
        if len(input_shape) != 3 or len(patch_shape) != 3:
            raise ValueError("Input and patch shapes must each contain three dimensions.")
        if any(value <= 0 for value in (*input_shape, *patch_shape)):
            raise ValueError("Input and patch dimensions must be greater than zero.")
        if input_channels <= 0 or embedding_dim <= 0 or depth <= 0 or heads <= 0 or head_dim <= 0:
            raise ValueError("Model dimensions, channels, depth, and heads must be positive.")
        if embedding_dim % heads != 0:
            raise ValueError("The embedding dimension must be divisible by the attention heads.")
        if mlp_ratio <= 0:
            raise ValueError("The transformer MLP ratio must be greater than zero.")
        for name, value in (
            ("attention dropout", attention_dropout),
            ("projection dropout", projection_dropout),
            ("head dropout", head_dropout),
        ):
            if not 0.0 <= value < 1.0:
                raise ValueError(f"The {name} must lie in the interval [0, 1).")
        if any(size % patch != 0 for size, patch in zip(input_shape, patch_shape, strict=True)):
            raise ValueError("Every input dimension must be divisible by its patch dimension.")
        token_grid = tuple(
            size // patch for size, patch in zip(input_shape, patch_shape, strict=True)
        )
        token_count = token_grid[0] * token_grid[1] * token_grid[2]
        self.input_shape_dhw = input_shape
        self.input_channels = input_channels
        self.patch_embedding = nn.Conv3d(
            input_channels,
            embedding_dim,
            kernel_size=patch_shape,
            stride=patch_shape,
        )
        self.class_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))
        self.position_embedding = nn.Parameter(torch.zeros(1, token_count + 1, embedding_dim))
        self.embedding_dropout = nn.Dropout(projection_dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=heads,
            dim_feedforward=int(round(embedding_dim * mlp_ratio)),
            dropout=attention_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=depth,
            norm=nn.LayerNorm(embedding_dim),
            enable_nested_tensor=False,
        )
        self.head = nn.Sequential(
            nn.Linear(embedding_dim, head_dim),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(head_dim, 1),
        )
        self._initialize_parameters()

    def _initialize_parameters(self) -> None:
        nn.init.trunc_normal_(self.class_token, std=0.02)
        nn.init.trunc_normal_(self.position_embedding, std=0.02)
        nn.init.kaiming_normal_(self.patch_embedding.weight, mode="fan_out")
        if self.patch_embedding.bias is not None:
            nn.init.zeros_(self.patch_embedding.bias)
        for module in self.head:
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                nn.init.zeros_(module.bias)

    def forward_features(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim != 5 or image.shape[1] != self.input_channels:
            raise ValueError(
                f"Expected a five-dimensional tensor with {self.input_channels} channels."
            )
        if tuple(image.shape[2:]) != self.input_shape_dhw:
            raise ValueError(
                f"Expected input shape {self.input_shape_dhw}, received {tuple(image.shape[2:])}."
            )
        tokens = self.patch_embedding(image).flatten(2).transpose(1, 2)
        class_token = self.class_token.expand(image.shape[0], -1, -1)
        tokens = torch.cat([class_token, tokens], dim=1)
        tokens = self.embedding_dropout(tokens + self.position_embedding)
        return self.encoder(tokens)[:, 0]

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(image)).squeeze(-1)


class ResidualBlock3D(nn.Module):
    """Basic residual block used by the anisotropic 3D ResNet-18 comparator."""

    expansion = 1

    def __init__(self, input_channels: int, output_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv3d(
            input_channels,
            output_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.norm1 = nn.BatchNorm3d(output_channels)
        self.activation = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(
            output_channels,
            output_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.norm2 = nn.BatchNorm3d(output_channels)
        if stride != 1 or input_channels != output_channels:
            self.skip = nn.Sequential(
                nn.Conv3d(
                    input_channels,
                    output_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm3d(output_channels),
            )
        else:
            self.skip = nn.Identity()

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        identity = self.skip(value)
        value = self.activation(self.norm1(self.conv1(value)))
        value = self.norm2(self.conv2(value))
        return self.activation(value + identity)


class ResNet3DSurvival(nn.Module):
    """Four-channel anisotropic 3D ResNet-18 with the shared Cox head design."""

    def __init__(
        self,
        *,
        input_channels: int = 4,
        stage_blocks: Sequence[int] = (2, 2, 2, 2),
        stage_channels: Sequence[int] = (64, 128, 256, 512),
        head_dim: int = 64,
        head_dropout: float = 0.30,
    ) -> None:
        super().__init__()
        if len(stage_blocks) != 4 or len(stage_channels) != 4:
            raise ValueError("The 3D ResNet requires four residual stages.")
        if input_channels <= 0 or head_dim <= 0:
            raise ValueError("Input channels and head dimension must be greater than zero.")
        if any(int(value) <= 0 for value in (*stage_blocks, *stage_channels)):
            raise ValueError("Residual block counts and channel widths must be greater than zero.")
        if not 0.0 <= head_dropout < 1.0:
            raise ValueError("The head dropout must lie in the interval [0, 1).")
        self.input_channels = input_channels
        self.stem = nn.Sequential(
            nn.Conv3d(
                input_channels,
                stage_channels[0],
                kernel_size=(3, 7, 7),
                stride=(1, 2, 2),
                padding=(1, 3, 3),
                bias=False,
            ),
            nn.BatchNorm3d(stage_channels[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
        )
        stages = []
        current_channels = stage_channels[0]
        for stage_index, (blocks, output_channels) in enumerate(
            zip(stage_blocks, stage_channels, strict=True)
        ):
            stride = 1 if stage_index == 0 else 2
            layers = [ResidualBlock3D(current_channels, output_channels, stride=stride)]
            layers.extend(
                ResidualBlock3D(output_channels, output_channels) for _ in range(blocks - 1)
            )
            stages.append(nn.Sequential(*layers))
            current_channels = output_channels
        self.stages = nn.Sequential(*stages)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.head = nn.Sequential(
            nn.Linear(stage_channels[-1], head_dim),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(head_dim, 1),
        )
        self._initialize_parameters()

    def _initialize_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv3d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm3d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                nn.init.zeros_(module.bias)

    def forward_features(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim != 5 or image.shape[1] != self.input_channels:
            raise ValueError(
                f"Expected a five-dimensional tensor with {self.input_channels} channels."
            )
        value = self.stages(self.stem(image))
        return self.pool(value).flatten(1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(image)).squeeze(-1)


def build_deep_survival_model(
    model_name: str,
    settings: dict[str, object],
    *,
    target_shape_dhw: Sequence[int],
) -> nn.Module:
    """Build one locked architecture from the study configuration."""

    input_channels = int(settings["input_channels"])
    if model_name == "vit":
        values = settings["vit"]
        if not isinstance(values, dict):
            raise ValueError("The ViT configuration is invalid.")
        return VisionTransformer3DSurvival(
            input_shape_dhw=target_shape_dhw,
            input_channels=input_channels,
            patch_shape_dhw=values["patch_shape_dhw"],
            embedding_dim=int(values["embedding_dim"]),
            depth=int(values["depth"]),
            heads=int(values["heads"]),
            mlp_ratio=float(values["mlp_ratio"]),
            attention_dropout=float(values["attention_dropout"]),
            projection_dropout=float(values["projection_dropout"]),
            head_dim=int(values["head_dim"]),
            head_dropout=float(values["head_dropout"]),
        )
    if model_name == "resnet":
        values = settings["resnet"]
        if not isinstance(values, dict):
            raise ValueError("The ResNet configuration is invalid.")
        return ResNet3DSurvival(
            input_channels=input_channels,
            stage_blocks=values["stage_blocks"],
            stage_channels=values["stage_channels"],
            head_dim=int(values["head_dim"]),
            head_dropout=float(values["head_dropout"]),
        )
    raise ValueError(f"Unknown deep survival model: {model_name}")
