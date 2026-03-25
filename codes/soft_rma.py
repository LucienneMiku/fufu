from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F


@dataclass
class SoftRMAConfig:
    enabled: bool = False
    apply_stages: List[int] = None
    gaussian_kernel_size: int = 9
    gaussian_sigma: float = 1.8
    boundary_gamma: float = 1.2
    relax_strength: float = 0.75
    min_boundary_strength: float = 0.15
    hard_block_bias: float = -18.0
    soft_block_bias: float = -4.0
    conservative_cross_region_only: bool = True
    key_boundary_only: bool = True
    include_unowned_region: bool = True
    soften_cross_ref_keys: bool = True
    cross_ref_relax_scale: float = 0.35
    soften_prompt_keys: bool = True
    prompt_relax_scale: float = 0.18
    stage_relax_scale: Dict[str, float] = None

    @classmethod
    def from_dict(cls, config: Optional[Dict[str, Any]]) -> "SoftRMAConfig":
        config = config or {}
        apply_stages = config.get("apply_stages", [1])
        if not isinstance(apply_stages, list):
            apply_stages = [1]
        cfg = cls(
            enabled=bool(config.get("enabled", False)),
            apply_stages=[int(x) for x in apply_stages],
            gaussian_kernel_size=max(3, int(config.get("gaussian_kernel_size", 9))),
            gaussian_sigma=max(1e-4, float(config.get("gaussian_sigma", 1.8))),
            boundary_gamma=max(1e-4, float(config.get("boundary_gamma", 1.2))),
            relax_strength=min(1.0, max(0.0, float(config.get("relax_strength", 0.75)))),
            min_boundary_strength=min(1.0, max(0.0, float(config.get("min_boundary_strength", 0.15)))),
            hard_block_bias=float(config.get("hard_block_bias", -18.0)),
            soft_block_bias=float(config.get("soft_block_bias", -4.0)),
            conservative_cross_region_only=bool(config.get("conservative_cross_region_only", True)),
            key_boundary_only=bool(config.get("key_boundary_only", True)),
            include_unowned_region=bool(config.get("include_unowned_region", True)),
            soften_cross_ref_keys=bool(config.get("soften_cross_ref_keys", True)),
            cross_ref_relax_scale=min(1.0, max(0.0, float(config.get("cross_ref_relax_scale", 0.35)))),
            soften_prompt_keys=bool(config.get("soften_prompt_keys", True)),
            prompt_relax_scale=min(1.0, max(0.0, float(config.get("prompt_relax_scale", 0.18)))),
            stage_relax_scale={
                str(k): float(v)
                for k, v in (config.get("stage_relax_scale", {"1": 1.0, "2": 0.55}) or {}).items()
            },
        )
        if cfg.gaussian_kernel_size % 2 == 0:
            cfg.gaussian_kernel_size += 1
        if cfg.soft_block_bias < cfg.hard_block_bias:
            cfg.soft_block_bias = cfg.hard_block_bias
        return cfg


class SoftRMAProcessor:
    """Converts hard binary RMA masks into smooth additive attention bias masks near region boundaries."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = SoftRMAConfig.from_dict(config)
        self.last_stats: Dict[str, float] = {}

    def is_active_for_stage(self, stage: Optional[int]) -> bool:
        if not self.config.enabled:
            return False
        if stage is None:
            return True
        return int(stage) in set(self.config.apply_stages)

    def _stage_relax_scale(self, stage: Optional[int]) -> float:
        if stage is None:
            return 1.0
        val = self.config.stage_relax_scale.get(str(int(stage)), 1.0)
        return min(1.0, max(0.0, float(val)))

    def apply(
        self,
        hard_mask: torch.Tensor,
        processed_inputs: Dict[str, Any],
        stage: Optional[int] = None,
    ) -> torch.Tensor:
        if not self.is_active_for_stage(stage):
            return hard_mask

        hard_bool = hard_mask.to(torch.bool)
        base_bias = torch.full_like(hard_bool, fill_value=self.config.hard_block_bias, dtype=torch.float32)
        base_bias[hard_bool] = 0.0
        stage_scale = self._stage_relax_scale(stage)

        boundary_pack = self._build_boundary_softness_and_owner(processed_inputs)
        if boundary_pack is None:
            self.last_stats = {
                "enabled": 1.0,
                "stage": float(stage if stage is not None else -1),
                "stage_scale": float(stage_scale),
                "softened_ratio": 0.0,
                "boundary_token_ratio": 0.0,
            }
            return base_bias.to(torch.float16)

        soften_strength, owner_ids = boundary_pack

        prompt_tokens_num = int(processed_inputs["prompt_tokens_num"])
        latent_tokens_num = int(processed_inputs["latent_tokens_num"])
        latent_start = prompt_tokens_num
        latent_end = prompt_tokens_num + latent_tokens_num

        sub_bias = base_bias[latent_start:latent_end, latent_start:latent_end]
        sub_hard = hard_bool[latent_start:latent_end, latent_start:latent_end]

        is_blocked = ~sub_hard
        owner_i = owner_ids[:, None]
        owner_j = owner_ids[None, :]

        if self.config.include_unowned_region:
            # -1 denotes unowned/background latent tokens; allow conservative blending with neighboring owned regions.
            valid_pair = (owner_i >= -1) & (owner_j >= -1)
        else:
            valid_pair = (owner_i >= 0) & (owner_j >= 0)
        cross_region = valid_pair & (owner_i != owner_j)
        if self.config.conservative_cross_region_only:
            candidate_block = is_blocked & cross_region
        else:
            candidate_block = is_blocked & valid_pair

        if self.config.key_boundary_only:
            boundary_gate = soften_strength[None, :]
        else:
            boundary_gate = torch.maximum(soften_strength[:, None], soften_strength[None, :])

        boundary_mask = boundary_gate >= float(self.config.min_boundary_strength)
        apply_soft_mask = candidate_block & boundary_mask

        relax = boundary_gate * float(self.config.relax_strength) * float(stage_scale)
        soft_bias = self.config.hard_block_bias + (
            self.config.soft_block_bias - self.config.hard_block_bias
        ) * relax

        updated_sub = torch.where(apply_soft_mask, soft_bias, sub_bias)
        base_bias[latent_start:latent_end, latent_start:latent_end] = updated_sub

        ref_softened = self._soften_cross_ref_keys(
            base_bias=base_bias,
            hard_bool=hard_bool,
            processed_inputs=processed_inputs,
            owner_ids=owner_ids,
            soften_strength=soften_strength,
            stage_scale=stage_scale,
            prompt_tokens_num=prompt_tokens_num,
        )
        prompt_softened = self._soften_prompt_keys(
            base_bias=base_bias,
            hard_bool=hard_bool,
            processed_inputs=processed_inputs,
            owner_ids=owner_ids,
            soften_strength=soften_strength,
            stage_scale=stage_scale,
            prompt_tokens_num=prompt_tokens_num,
        )

        total_blocked = int(is_blocked.sum().item())
        softened = int(apply_soft_mask.sum().item())
        boundary_tokens = int((soften_strength >= float(self.config.min_boundary_strength)).sum().item())
        self.last_stats = {
            "enabled": 1.0,
            "stage": float(stage if stage is not None else -1),
            "stage_scale": float(stage_scale),
            "softened_ratio": float(softened / total_blocked) if total_blocked > 0 else 0.0,
            "boundary_token_ratio": float(boundary_tokens / latent_tokens_num) if latent_tokens_num > 0 else 0.0,
            "softened_pairs": float(softened),
            "blocked_pairs": float(total_blocked),
            "ref_softened_pairs": float(ref_softened),
            "prompt_softened_pairs": float(prompt_softened),
        }
        return base_bias.to(torch.float16)

    def _soften_cross_ref_keys(
        self,
        base_bias: torch.Tensor,
        hard_bool: torch.Tensor,
        processed_inputs: Dict[str, Any],
        owner_ids: torch.Tensor,
        soften_strength: torch.Tensor,
        stage_scale: float,
        prompt_tokens_num: int,
    ) -> int:
        if not self.config.soften_cross_ref_keys:
            return 0

        region_index_list = processed_inputs.get("region_index_list", [])
        ref_img_index_list = processed_inputs.get("ref_img_index_list", [])
        token_indices = processed_inputs.get("token_indices", {})
        if len(region_index_list) == 0 or len(ref_img_index_list) == 0:
            return 0

        # Assume region i corresponds to ref image i in the standard LAMIC construction order.
        pair_count = min(len(region_index_list), len(ref_img_index_list))
        if pair_count <= 1:
            return 0

        boundary_query_local = torch.where(soften_strength >= float(self.config.min_boundary_strength))[0]
        if boundary_query_local.numel() == 0:
            return 0

        relaxed_pairs = 0
        for local_region_id in range(pair_count):
            # Boundary queries that belong to this region only.
            q_local = boundary_query_local[owner_ids[boundary_query_local] == local_region_id]
            if q_local.numel() == 0:
                continue

            q_global = q_local + prompt_tokens_num
            q_relax = soften_strength[q_local].to(torch.float32) * float(self.config.relax_strength)
            q_relax = q_relax * float(stage_scale) * float(self.config.cross_ref_relax_scale)
            q_relax = q_relax[:, None]

            for other_region_id in range(pair_count):
                if other_region_id == local_region_id:
                    continue
                ref_block_idx = ref_img_index_list[other_region_id]
                if ref_block_idx not in token_indices:
                    continue
                k_global = token_indices[ref_block_idx].to(torch.long)
                if k_global.numel() == 0:
                    continue

                # Only relax entries that are currently blocked.
                blocked = ~hard_bool[q_global[:, None], k_global[None, :]]
                if not bool(blocked.any()):
                    continue

                soft_bias_q = self.config.hard_block_bias + (
                    self.config.soft_block_bias - self.config.hard_block_bias
                ) * q_relax
                current = base_bias[q_global[:, None], k_global[None, :]]
                updated = torch.where(blocked, soft_bias_q.expand_as(current), current)
                base_bias[q_global[:, None], k_global[None, :]] = updated
                relaxed_pairs += int(blocked.sum().item())

        return relaxed_pairs

    def _soften_prompt_keys(
        self,
        base_bias: torch.Tensor,
        hard_bool: torch.Tensor,
        processed_inputs: Dict[str, Any],
        owner_ids: torch.Tensor,
        soften_strength: torch.Tensor,
        stage_scale: float,
        prompt_tokens_num: int,
    ) -> int:
        if not self.config.soften_prompt_keys:
            return 0

        prompt_index_list = processed_inputs.get("prompt_index_list", [])
        token_indices = processed_inputs.get("token_indices", {})
        if len(prompt_index_list) == 0:
            return 0

        prompt_keys = []
        for idx in prompt_index_list:
            if idx in token_indices:
                prompt_keys.append(token_indices[idx].to(torch.long))
        if len(prompt_keys) == 0:
            return 0
        k_global = torch.cat(prompt_keys, dim=0).unique()
        if k_global.numel() == 0:
            return 0

        boundary_query_local = torch.where(soften_strength >= float(self.config.min_boundary_strength))[0]
        if boundary_query_local.numel() == 0:
            return 0

        # Only owned region queries participate to avoid broad background leakage.
        q_local = boundary_query_local[owner_ids[boundary_query_local] >= 0]
        if q_local.numel() == 0:
            return 0

        q_global = q_local + prompt_tokens_num
        q_relax = soften_strength[q_local].to(torch.float32) * float(self.config.relax_strength)
        q_relax = q_relax * float(stage_scale) * float(self.config.prompt_relax_scale)
        q_relax = q_relax[:, None]

        blocked = ~hard_bool[q_global[:, None], k_global[None, :]]
        if not bool(blocked.any()):
            return 0

        soft_bias_q = self.config.hard_block_bias + (
            self.config.soft_block_bias - self.config.hard_block_bias
        ) * q_relax
        current = base_bias[q_global[:, None], k_global[None, :]]
        updated = torch.where(blocked, soft_bias_q.expand_as(current), current)
        base_bias[q_global[:, None], k_global[None, :]] = updated
        return int(blocked.sum().item())

    def _build_boundary_softness_and_owner(
        self, processed_inputs: Dict[str, Any]
    ) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        packed_h = int(processed_inputs.get("packed_latent_height", 0))
        packed_w = int(processed_inputs.get("packed_latent_width", 0))
        latent_tokens_num = int(processed_inputs.get("latent_tokens_num", 0))
        if packed_h <= 0 or packed_w <= 0 or packed_h * packed_w != latent_tokens_num:
            return None

        token_owner = torch.full((latent_tokens_num,), fill_value=-1, dtype=torch.long)
        region_index_list = processed_inputs.get("region_index_list", [])
        mitigated = processed_inputs.get("mitigated_indices", {})
        prompt_tokens_num = int(processed_inputs.get("prompt_tokens_num", 0))

        valid_region_count = 0
        for local_region_id, global_region_idx in enumerate(region_index_list):
            token_indices = mitigated.get(global_region_idx, processed_inputs["token_indices"][global_region_idx])
            local_latent_ids = token_indices.to(torch.long) - prompt_tokens_num
            keep = (local_latent_ids >= 0) & (local_latent_ids < latent_tokens_num)
            local_latent_ids = local_latent_ids[keep]
            if local_latent_ids.numel() == 0:
                continue
            token_owner[local_latent_ids] = local_region_id
            valid_region_count += 1

        if valid_region_count <= 1:
            return None

        owner_map = token_owner.view(packed_h, packed_w)
        boundary = torch.zeros((packed_h, packed_w), dtype=torch.float32)

        # 4-neighborhood boundary detection between different region IDs.
        if packed_h > 1:
            up = owner_map[1:, :]
            down = owner_map[:-1, :]
            diff = (up != down) & (up >= 0) & (down >= 0)
            boundary[1:, :] = torch.maximum(boundary[1:, :], diff.to(torch.float32))
            boundary[:-1, :] = torch.maximum(boundary[:-1, :], diff.to(torch.float32))
        if packed_w > 1:
            left = owner_map[:, 1:]
            right = owner_map[:, :-1]
            diff = (left != right) & (left >= 0) & (right >= 0)
            boundary[:, 1:] = torch.maximum(boundary[:, 1:], diff.to(torch.float32))
            boundary[:, :-1] = torch.maximum(boundary[:, :-1], diff.to(torch.float32))

        if float(boundary.max().item()) <= 0.0:
            return torch.zeros((latent_tokens_num,), dtype=torch.float32), token_owner

        smooth = self._gaussian_blur(boundary)
        if float(smooth.max().item()) > 0.0:
            smooth = smooth / smooth.max()
        smooth = torch.pow(smooth.clamp(0.0, 1.0), float(self.config.boundary_gamma))
        return smooth.reshape(-1), token_owner

    def _gaussian_blur(self, x: torch.Tensor) -> torch.Tensor:
        kernel = self._gaussian_kernel(
            self.config.gaussian_kernel_size,
            self.config.gaussian_sigma,
            device=x.device,
            dtype=x.dtype,
        )
        pad = self.config.gaussian_kernel_size // 2
        out = F.conv2d(x[None, None, ...], kernel, padding=pad)
        return out[0, 0]

    @staticmethod
    def _gaussian_kernel(size: int, sigma: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        radius = size // 2
        coords = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
        xx, yy = torch.meshgrid(coords, coords, indexing="ij")
        kernel = torch.exp(-(xx * xx + yy * yy) / (2.0 * sigma * sigma))
        kernel = kernel / kernel.sum().clamp_min(1e-12)
        return kernel[None, None, :, :]
