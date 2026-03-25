from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F


@dataclass
class DynamicMaskEvolutionConfig:
    enabled: bool = False
    threshold: float = 0.35
    prior_weight: float = 0.4
    expand_pixels: int = 1
    min_retention_ratio: float = 0.3
    max_region_area_ratio: float = 0.6
    min_attention_std: float = 0.015
    max_change_ratio: float = 0.75
    min_iou_with_prior: float = 0.45
    use_topk_selection: bool = True
    target_area_scale: float = 1.0
    max_area_growth_ratio: float = 0.15
    min_area_ratio_vs_prior: float = 0.85
    max_centroid_shift_ratio: float = 0.2
    allow_region_ids: Optional[List[int]] = None
    block_region_ids: Optional[List[int]] = None
    max_updated_regions: int = 1


class DynamicMaskEvolutionController:
    """Collects early-step cross-attention statistics and evolves region token masks."""

    def __init__(self, config: dict, processed_inputs: dict):
        cfg = DynamicMaskEvolutionConfig(
            enabled=bool(config.get("enabled", False)),
            threshold=float(config.get("threshold", 0.35)),
            prior_weight=float(config.get("prior_weight", 0.4)),
            expand_pixels=int(config.get("expand_pixels", 1)),
            min_retention_ratio=float(config.get("min_retention_ratio", 0.3)),
            max_region_area_ratio=float(config.get("max_region_area_ratio", 0.6)),
            min_attention_std=float(config.get("min_attention_std", 0.015)),
            max_change_ratio=float(config.get("max_change_ratio", 0.75)),
            min_iou_with_prior=float(config.get("min_iou_with_prior", 0.45)),
            use_topk_selection=bool(config.get("use_topk_selection", True)),
            target_area_scale=float(config.get("target_area_scale", 1.0)),
            max_area_growth_ratio=float(config.get("max_area_growth_ratio", 0.15)),
            min_area_ratio_vs_prior=float(config.get("min_area_ratio_vs_prior", 0.85)),
            max_centroid_shift_ratio=float(config.get("max_centroid_shift_ratio", 0.2)),
            allow_region_ids=[int(x) for x in config.get("allow_region_ids", [])] if config.get("allow_region_ids") is not None else None,
            block_region_ids=[int(x) for x in config.get("block_region_ids", [])] if config.get("block_region_ids") is not None else None,
            max_updated_regions=max(0, int(config.get("max_updated_regions", 1))),
        )
        self.config = cfg
        self.processed_inputs = processed_inputs
        self.ref_num = int(processed_inputs["ref_num"])
        self.prompt_tokens_num = int(processed_inputs["prompt_tokens_num"])
        self.latent_tokens_num = int(processed_inputs["latent_tokens_num"])

        self.region_indices: List[int] = [processed_inputs["latents_start_index"] + i for i in range(self.ref_num)]
        self.prompt_token_ids: Dict[int, torch.Tensor] = {
            region_idx: processed_inputs["token_indices"][i].to(torch.long)
            for i, region_idx in enumerate(self.region_indices)
        }

        self.score_sum: Dict[int, Optional[torch.Tensor]] = {idx: None for idx in self.region_indices}
        self.score_count: Dict[int, int] = {idx: 0 for idx in self.region_indices}
        self.last_report: Dict[int, Dict[str, float | str]] = {}

    def consume_query_key(self, query: torch.Tensor, key: torch.Tensor, context_len: int) -> None:
        if not self.config.enabled:
            return

        if context_len <= 0 or self.latent_tokens_num <= 0:
            return

        # Query/Key are [B, H, S, D]. We only track generated latent tokens.
        latent_end = min(context_len + self.latent_tokens_num, query.shape[2])
        if latent_end <= context_len:
            return

        img_query = query[:, :, context_len:latent_end, :]
        ctx_key = key[:, :, :context_len, :]
        # Normalize attention over all context tokens first, then aggregate entity token mass.
        scores_all = torch.einsum("bhid,bhjd->bhij", img_query.float(), ctx_key.float())
        probs_all = torch.softmax(scores_all / (img_query.shape[-1] ** 0.5), dim=-1)

        for region_idx in self.region_indices:
            token_ids = self.prompt_token_ids[region_idx]
            if token_ids.numel() == 0:
                continue

            valid_ids = token_ids[token_ids < context_len]
            if valid_ids.numel() == 0:
                continue

            # Per-image-token attention mass assigned to this entity's prompt tokens.
            region_mass = probs_all[:, :, :, valid_ids].mean(dim=-1)
            region_score = region_mass.mean(dim=1).mean(dim=0).detach().cpu()

            if self.score_sum[region_idx] is None:
                self.score_sum[region_idx] = region_score
            else:
                self.score_sum[region_idx] = self.score_sum[region_idx] + region_score
            self.score_count[region_idx] += 1

    def evolve_region_token_indices(self, device: torch.device) -> Dict[int, torch.Tensor]:
        if not self.config.enabled:
            return {}

        width = int(self.processed_inputs["output_width"] // 16)
        height = int(self.processed_inputs["output_height"] // 16)
        if width * height != self.latent_tokens_num:
            return {}

        evolved_candidates: Dict[int, torch.Tensor] = {}
        confidence_scores: Dict[int, float] = {}
        report: Dict[int, Dict[str, float | str]] = {}
        for region_idx in self.region_indices:
            local_region_id = int(region_idx - self.processed_inputs["latents_start_index"])
            if self.config.allow_region_ids is not None and local_region_id not in self.config.allow_region_ids:
                report[region_idx] = {
                    "status": "skip_not_allowed_region",
                    "region_id": float(local_region_id),
                }
                continue
            if self.config.block_region_ids is not None and local_region_id in self.config.block_region_ids:
                report[region_idx] = {
                    "status": "skip_blocked_region",
                    "region_id": float(local_region_id),
                }
                continue

            if self.score_count[region_idx] == 0 or self.score_sum[region_idx] is None:
                report[region_idx] = {"status": "skip_no_attn_stats"}
                continue

            avg_score = self.score_sum[region_idx] / float(self.score_count[region_idx])
            avg_score = self._normalize(avg_score)
            attn_std = float(torch.std(avg_score).item())
            if attn_std < self.config.min_attention_std:
                # Attention is too flat to form a reliable pseudo-mask.
                report[region_idx] = {
                    "status": "skip_low_attention_std",
                    "attention_std": attn_std,
                }
                continue

            prior_tokens = self.processed_inputs["token_indices"][region_idx].to(torch.long)
            if prior_tokens.numel() == 0:
                report[region_idx] = {"status": "skip_empty_prior"}
                continue

            region_area_ratio = float(prior_tokens.numel()) / float(self.latent_tokens_num)
            if region_area_ratio > self.config.max_region_area_ratio:
                # For global/full-image regions, dynamic evolution is unstable and often hurts quality.
                report[region_idx] = {
                    "status": "skip_large_region",
                    "region_area_ratio": region_area_ratio,
                }
                continue

            prior_mask = torch.zeros(self.latent_tokens_num, dtype=torch.float32)
            prior_mask[prior_tokens.clamp(0, self.latent_tokens_num - 1)] = 1.0

            combined = (1.0 - self.config.prior_weight) * avg_score + self.config.prior_weight * prior_mask

            prior_mask_2d = prior_mask.view(1, 1, height, width)
            search_area_2d = torch.ones_like(prior_mask_2d)

            if self.config.expand_pixels > 0:
                k = 2 * self.config.expand_pixels + 1
                search_area_2d = F.max_pool2d(prior_mask_2d, kernel_size=k, stride=1, padding=self.config.expand_pixels)
            search_area = (search_area_2d.view(-1) > 0).to(torch.float32)

            if self.config.use_topk_selection:
                prior_count = int(prior_tokens.numel())
                min_keep = max(1, int(prior_count * self.config.min_area_ratio_vs_prior))
                max_keep = max(min_keep, int(prior_count * (1.0 + self.config.max_area_growth_ratio)))
                target_keep = int(round(prior_count * self.config.target_area_scale))
                target_keep = max(min_keep, min(max_keep, target_keep))

                candidate_scores = combined * search_area + 1e-4 * prior_mask
                if float(candidate_scores.sum().item()) <= 0:
                    evolved_flat = prior_mask.clone()
                else:
                    topk = torch.topk(candidate_scores, k=min(target_keep, candidate_scores.numel())).indices
                    evolved_flat = torch.zeros_like(prior_mask)
                    evolved_flat[topk] = 1.0
            else:
                evolved_mask = (combined >= self.config.threshold).to(torch.float32)
                evolved_flat = evolved_mask * search_area

            min_keep = max(1, int(prior_tokens.numel() * self.config.min_retention_ratio))
            if int(evolved_flat.sum().item()) < min_keep:
                evolved_flat = prior_mask

            evolved_indices = torch.nonzero(evolved_flat > 0.5, as_tuple=False).flatten()
            change_ratio = self._compute_change_ratio(prior_tokens, evolved_indices)
            if change_ratio > self.config.max_change_ratio:
                # Avoid abrupt mask jumps that typically cause deformed generations.
                report[region_idx] = {
                    "status": "skip_large_change",
                    "change_ratio": change_ratio,
                }
                continue

            iou_with_prior = self._compute_iou(prior_tokens, evolved_indices)
            if iou_with_prior < self.config.min_iou_with_prior:
                report[region_idx] = {
                    "status": "skip_low_iou",
                    "iou_with_prior": iou_with_prior,
                }
                continue

            centroid_shift_ratio = self._compute_centroid_shift_ratio(prior_tokens, evolved_indices, width, height)
            if centroid_shift_ratio > self.config.max_centroid_shift_ratio:
                report[region_idx] = {
                    "status": "skip_large_centroid_shift",
                    "centroid_shift_ratio": centroid_shift_ratio,
                }
                continue

            evolved_candidates[region_idx] = evolved_indices.to(device)
            confidence_scores[region_idx] = float(attn_std * max(0.0, iou_with_prior) / (1.0 + max(0.0, change_ratio)))
            report[region_idx] = {
                "status": "applied",
                "attention_std": attn_std,
                "region_area_ratio": region_area_ratio,
                "change_ratio": change_ratio,
                "iou_with_prior": iou_with_prior,
                "centroid_shift_ratio": centroid_shift_ratio,
                "new_tokens": float(evolved_indices.numel()),
            }

        evolved: Dict[int, torch.Tensor] = {}
        if self.config.max_updated_regions > 0 and len(evolved_candidates) > self.config.max_updated_regions:
            sorted_region_ids = sorted(
                evolved_candidates.keys(), key=lambda rid: confidence_scores.get(rid, 0.0), reverse=True
            )
            keep_ids = set(sorted_region_ids[: self.config.max_updated_regions])
            for rid in evolved_candidates.keys():
                if rid in keep_ids:
                    evolved[rid] = evolved_candidates[rid]
                else:
                    report[rid] = {
                        "status": "skip_low_global_confidence",
                        "confidence": confidence_scores.get(rid, 0.0),
                    }
        else:
            evolved = evolved_candidates

        self.last_report = report
        return evolved

    @staticmethod
    def _normalize(x: torch.Tensor) -> torch.Tensor:
        x_min = torch.min(x)
        x_max = torch.max(x)
        if torch.abs(x_max - x_min) < 1e-6:
            return torch.zeros_like(x)
        return (x - x_min) / (x_max - x_min)

    @staticmethod
    def _compute_change_ratio(prior_tokens: torch.Tensor, evolved_tokens: torch.Tensor) -> float:
        if prior_tokens.numel() == 0:
            return 0.0
        prior_set = set(prior_tokens.tolist())
        evolved_set = set(evolved_tokens.tolist())
        symmetric_diff = len(prior_set.symmetric_difference(evolved_set))
        return float(symmetric_diff) / float(max(1, len(prior_set)))

    @staticmethod
    def _compute_iou(prior_tokens: torch.Tensor, evolved_tokens: torch.Tensor) -> float:
        prior_set = set(prior_tokens.tolist())
        evolved_set = set(evolved_tokens.tolist())
        if not prior_set and not evolved_set:
            return 1.0
        if not prior_set or not evolved_set:
            return 0.0
        intersection = len(prior_set.intersection(evolved_set))
        union = len(prior_set.union(evolved_set))
        return float(intersection) / float(max(1, union))

    @staticmethod
    def _compute_centroid_shift_ratio(
        prior_tokens: torch.Tensor, evolved_tokens: torch.Tensor, width: int, height: int
    ) -> float:
        if prior_tokens.numel() == 0 or evolved_tokens.numel() == 0:
            return 0.0

        prior_xy = torch.stack((prior_tokens % width, prior_tokens // width), dim=-1).to(torch.float32)
        evolved_xy = torch.stack((evolved_tokens % width, evolved_tokens // width), dim=-1).to(torch.float32)
        prior_center = prior_xy.mean(dim=0)
        evolved_center = evolved_xy.mean(dim=0)

        shift = torch.norm(evolved_center - prior_center, p=2)
        diag = (width**2 + height**2) ** 0.5
        if diag <= 1e-6:
            return 0.0
        return float((shift / diag).item())
