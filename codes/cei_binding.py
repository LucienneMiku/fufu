from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Dict, List, Tuple

import torch


def _normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _strip_articles(text: str) -> str:
    return re.sub(r"^(a|an|the)\s+", "", text.strip(), flags=re.IGNORECASE)


@dataclass
class CEIBindingConfig:
    enabled: bool = False
    use_prompt_when_cei_empty: bool = True
    include_sad_desc_context: bool = False
    strong_keywords: List[str] | None = None
    weak_keywords: List[str] | None = None
    strong_threshold: float = 1.0
    weak_threshold: float = -1.0
    max_pairs_to_adjust: int = 8
    relation_window_tokens: int = 8
    allow_scene_strong_pairs: bool = False
    nearest_entity_fallback_window: int = 16
    disable_suppress_when_no_pairs: bool = True
    merge_prompt_with_cei: bool = True
    apply_stage1_strong: bool = True
    apply_stage1_weak: bool = True
    apply_stage2_strong: bool = True
    apply_stage2_weak: bool = False
    fallback_to_sad_desc_when_no_pairs: bool = True
    fallback_to_prompt_when_cei_has_no_entity: bool = True
    assume_single_pair_when_cei_relation_only: bool = True
    single_pair_fallback_apply_stage1: bool = False
    single_pair_fallback_apply_stage2: bool = True
    suppress_scene_pairs_only: bool = True
    preserve_human_scene_links: bool = True
    preserve_uncontrolled_region_links: bool = True
    suppress_non_strong_stage1: bool = False
    suppress_non_strong_stage2: bool = False
    disable_weak_when_no_strong: bool = True
    require_strong_pairs_for_suppression: bool = True
    prune_nonhuman_pairs_with_human_anchor: bool = True
    preserve_human_linked_scene_links: bool = True


class CEIBindingProcessor:
    """Lightweight CEI parser and region-pair mask controller.

    This module is designed as a plug-in preprocessor. It can be replaced by an
    LLM parser later while preserving the same output contract.
    """

    def __init__(self, config: dict):
        default_strong = [
            "ride",
            "riding",
            "hold",
            "holding",
            "hug",
            "hugging",
            "kiss",
            "wear",
            "wearing",
            "dressed on",
            "carry",
            "carrying",
            "play with",
            "playing with",
            "sit on",
            "standing on",
            "touch",
            "interact",
            "next to",
        ]
        default_weak = [
            "left of",
            "on the left",
            "right of",
            "on the right",
            "in front of",
            "behind",
            "far from",
            "apart",
            "separate",
            "without touching",
        ]
        self.config = CEIBindingConfig(
            enabled=bool(config.get("enabled", False)),
            use_prompt_when_cei_empty=bool(config.get("use_prompt_when_cei_empty", True)),
            include_sad_desc_context=bool(config.get("include_sad_desc_context", False)),
            strong_keywords=list(config.get("strong_keywords", default_strong)),
            weak_keywords=list(config.get("weak_keywords", default_weak)),
            strong_threshold=float(config.get("strong_threshold", 1.0)),
            weak_threshold=float(config.get("weak_threshold", -1.0)),
            max_pairs_to_adjust=max(0, int(config.get("max_pairs_to_adjust", 8))),
            relation_window_tokens=max(1, int(config.get("relation_window_tokens", 8))),
            allow_scene_strong_pairs=bool(config.get("allow_scene_strong_pairs", False)),
            nearest_entity_fallback_window=max(1, int(config.get("nearest_entity_fallback_window", 16))),
            disable_suppress_when_no_pairs=bool(config.get("disable_suppress_when_no_pairs", True)),
            merge_prompt_with_cei=bool(config.get("merge_prompt_with_cei", True)),
            apply_stage1_strong=bool(config.get("apply_stage1", {}).get("strong", True)),
            apply_stage1_weak=bool(config.get("apply_stage1", {}).get("weak", True)),
            apply_stage2_strong=bool(config.get("apply_stage2", {}).get("strong", True)),
            apply_stage2_weak=bool(config.get("apply_stage2", {}).get("weak", False)),
            fallback_to_sad_desc_when_no_pairs=bool(config.get("fallback_to_sad_desc_when_no_pairs", True)),
            fallback_to_prompt_when_cei_has_no_entity=bool(config.get("fallback_to_prompt_when_cei_has_no_entity", True)),
            assume_single_pair_when_cei_relation_only=bool(config.get("assume_single_pair_when_cei_relation_only", True)),
            single_pair_fallback_apply_stage1=bool(config.get("single_pair_fallback_apply_stage1", False)),
            single_pair_fallback_apply_stage2=bool(config.get("single_pair_fallback_apply_stage2", True)),
            suppress_scene_pairs_only=bool(config.get("suppress_scene_pairs_only", True)),
            preserve_human_scene_links=bool(config.get("preserve_human_scene_links", True)),
            preserve_uncontrolled_region_links=bool(config.get("preserve_uncontrolled_region_links", True)),
            suppress_non_strong_stage1=bool(config.get("apply_stage1", {}).get("suppress_non_strong", False)),
            suppress_non_strong_stage2=bool(config.get("apply_stage2", {}).get("suppress_non_strong", False)),
            disable_weak_when_no_strong=bool(config.get("disable_weak_when_no_strong", True)),
            require_strong_pairs_for_suppression=bool(config.get("require_strong_pairs_for_suppression", True)),
            prune_nonhuman_pairs_with_human_anchor=bool(config.get("prune_nonhuman_pairs_with_human_anchor", True)),
            preserve_human_linked_scene_links=bool(config.get("preserve_human_linked_scene_links", True)),
        )
        self.last_report: Dict[str, object] = {}
        self.last_apply_stats: Dict[str, object] = {}

    def _build_entity_aliases(self, inputs: dict, processed_inputs: dict) -> Dict[int, List[str]]:
        aliases: Dict[int, List[str]] = {}
        ref_num = int(processed_inputs["ref_num"])
        region_start = int(processed_inputs["latents_start_index"])

        for i in range(1, ref_num + 1):
            region_idx = region_start + i - 1
            sad = inputs.get(f"ref_img_{i}", {}).get("SAD", {})
            if isinstance(sad, dict):
                entity_id = str(sad.get("id", "")).strip()
            else:
                entity_id = str(sad).strip()

            candidate_aliases = []
            if entity_id:
                candidate_aliases.append(entity_id)
                candidate_aliases.append(_strip_articles(entity_id))
                entity_id_clean = _normalize_text(_strip_articles(entity_id))
                entity_tokens = entity_id_clean.split()
                if len(entity_tokens) >= 1:
                    candidate_aliases.append(entity_tokens[-1])
                if len(entity_tokens) >= 2:
                    candidate_aliases.append(" ".join(entity_tokens[-2:]))

            # Fallback aliases for robust matching when SAD id is short/noisy.
            candidate_aliases.extend([f"ref img {i}", f"entity {i}", f"object {i}"])

            norm_aliases = []
            for alias in candidate_aliases:
                normalized = _normalize_text(alias)
                if normalized and normalized not in norm_aliases:
                    norm_aliases.append(normalized)

            aliases[region_idx] = norm_aliases

        return aliases

    def _build_context_text(self, inputs: dict, processed_inputs: dict, force_include_sad_desc: bool = False) -> str:
        cei_text = str(inputs.get("CEI", "") or "").strip()
        prompt_text = str(inputs.get("prompt", "") or "").strip()

        text_parts = []
        if cei_text and prompt_text and self.config.merge_prompt_with_cei:
            text_parts.append(cei_text)
            text_parts.append(prompt_text)
        elif cei_text:
            text_parts.append(cei_text)
        elif self.config.use_prompt_when_cei_empty and prompt_text:
            text_parts.append(prompt_text)

        if self.config.include_sad_desc_context or force_include_sad_desc:
            ref_num = int(processed_inputs["ref_num"])
            for i in range(1, ref_num + 1):
                sad = inputs.get(f"ref_img_{i}", {}).get("SAD", {})
                if isinstance(sad, dict):
                    desc = str(sad.get("desc", "") or "").strip()
                    if desc:
                        text_parts.append(desc)

        return " . ".join(text_parts)

    def _mention_entity(self, normalized_clause: str, aliases: List[str]) -> bool:
        for alias in aliases:
            if not alias:
                continue
            if re.search(rf"\b{re.escape(alias)}\b", normalized_clause):
                return True
        return False

    def _score_clause(self, normalized_clause: str) -> float:
        score = 0.0
        for kw in self.config.strong_keywords or []:
            if kw and kw in normalized_clause:
                score += 1.0
        for kw in self.config.weak_keywords or []:
            if kw and kw in normalized_clause:
                score -= 1.0
        return score

    @staticmethod
    def _find_phrase_positions(tokens: List[str], phrase: str) -> List[int]:
        phrase_tokens = [t for t in phrase.strip().split() if t]
        if not phrase_tokens or len(phrase_tokens) > len(tokens):
            return []
        out = []
        n = len(phrase_tokens)
        for i in range(0, len(tokens) - n + 1):
            if tokens[i : i + n] == phrase_tokens:
                out.append(i)
        return out

    def _score_pair_in_clause(self, tokens: List[str], aliases_a: List[str], aliases_b: List[str]) -> float:
        a_positions: List[int] = []
        b_positions: List[int] = []
        for alias in aliases_a:
            a_positions.extend(self._find_phrase_positions(tokens, alias))
        for alias in aliases_b:
            b_positions.extend(self._find_phrase_positions(tokens, alias))

        if len(a_positions) == 0 or len(b_positions) == 0:
            return 0.0

        score = 0.0
        win = int(self.config.relation_window_tokens)
        for kw in self.config.strong_keywords or []:
            kw_positions = self._find_phrase_positions(tokens, kw)
            for kp in kw_positions:
                near_a = any(abs(kp - ap) <= win for ap in a_positions)
                near_b = any(abs(kp - bp) <= win for bp in b_positions)
                if near_a and near_b:
                    score += 1.0

        for kw in self.config.weak_keywords or []:
            kw_positions = self._find_phrase_positions(tokens, kw)
            for kp in kw_positions:
                near_a = any(abs(kp - ap) <= win for ap in a_positions)
                near_b = any(abs(kp - bp) <= win for bp in b_positions)
                if near_a and near_b:
                    score -= 1.0

        return score

    def _score_clause_by_nearest_entities(
        self,
        tokens: List[str],
        region_ids: List[int],
        entity_aliases: Dict[int, List[str]],
        pair_scores: Dict[Tuple[int, int], float],
    ) -> None:
        if len(tokens) == 0 or len(region_ids) < 2:
            return

        entity_positions: Dict[int, List[int]] = {}
        for rid in region_ids:
            positions: List[int] = []
            for alias in entity_aliases[rid]:
                positions.extend(self._find_phrase_positions(tokens, alias))
            if positions:
                entity_positions[rid] = positions

        if len(entity_positions) < 2:
            return

        win = int(self.config.nearest_entity_fallback_window)
        strong_kw_positions: List[int] = []
        weak_kw_positions: List[int] = []
        for kw in self.config.strong_keywords or []:
            strong_kw_positions.extend(self._find_phrase_positions(tokens, kw))
        for kw in self.config.weak_keywords or []:
            weak_kw_positions.extend(self._find_phrase_positions(tokens, kw))

        def apply_kw_positions(kp_list: List[int], sign: float) -> None:
            for kp in kp_list:
                dists: List[Tuple[int, int]] = []
                for rid, pos_list in entity_positions.items():
                    min_dist = min(abs(kp - p) for p in pos_list)
                    if min_dist <= win:
                        dists.append((rid, min_dist))
                dists = sorted(dists, key=lambda x: x[1])
                if len(dists) < 2:
                    continue
                a = int(dists[0][0])
                b = int(dists[1][0])
                pair = tuple(sorted((a, b)))
                pair_scores[pair] = pair_scores.get(pair, 0.0) + sign

        apply_kw_positions(strong_kw_positions, 1.0)
        apply_kw_positions(weak_kw_positions, -1.0)

    def _extract_pairs(
        self,
        context_text: str,
        entity_aliases: Dict[int, List[str]],
        region_types: Dict[int, str],
    ) -> Tuple[List[Tuple[int, int, float]], List[Tuple[int, int, float]], Dict[str, float]]:
        normalized_text = _normalize_text(context_text)
        # Keep sentence-level context to avoid dropping shared subject in
        # subordinate clauses like "... while playing ...".
        clauses = [c.strip() for c in re.split(r"[\.;]", normalized_text) if c.strip()]

        region_ids = sorted(entity_aliases.keys())
        pair_scores: Dict[Tuple[int, int], float] = {}
        for idx_a in range(len(region_ids)):
            for idx_b in range(idx_a + 1, len(region_ids)):
                region_a = region_ids[idx_a]
                region_b = region_ids[idx_b]
                score = 0.0
                for clause in clauses:
                    tokens = [t for t in clause.split() if t]
                    score += self._score_pair_in_clause(tokens, entity_aliases[region_a], entity_aliases[region_b])

                type_a = region_types.get(region_a, "unknown")
                type_b = region_types.get(region_b, "unknown")
                if not self.config.allow_scene_strong_pairs and (type_a == "scene" or type_b == "scene") and score > 0.0:
                    score = 0.0
                pair_scores[(region_a, region_b)] = score

        # Suppress common false positives where two non-human entities are both
        # linked to the same human subject but not directly related to each other.
        if self.config.prune_nonhuman_pairs_with_human_anchor:
            human_regions = [rid for rid in region_ids if region_types.get(rid, "unknown") == "human"]
            if human_regions:
                for pair_key, score in list(pair_scores.items()):
                    if score < self.config.strong_threshold:
                        continue
                    region_a, region_b = pair_key
                    type_a = region_types.get(region_a, "unknown")
                    type_b = region_types.get(region_b, "unknown")
                    if type_a == "human" or type_b == "human":
                        continue
                    for human_region in human_regions:
                        pair_ha = tuple(sorted((human_region, region_a)))
                        pair_hb = tuple(sorted((human_region, region_b)))
                        score_ha = float(pair_scores.get(pair_ha, 0.0))
                        score_hb = float(pair_scores.get(pair_hb, 0.0))
                        if score_ha >= self.config.strong_threshold and score_hb >= self.config.strong_threshold:
                            pair_scores[pair_key] = 0.0
                            break

        # If exact pair scoring fails, use nearest-entity fallback around relation keywords.
        if all(abs(v) < 1e-9 for v in pair_scores.values()):
            for clause in clauses:
                tokens = [t for t in clause.split() if t]
                self._score_clause_by_nearest_entities(
                    tokens=tokens,
                    region_ids=region_ids,
                    entity_aliases=entity_aliases,
                    pair_scores=pair_scores,
                )

            # Re-apply scene strong-pair protection after fallback scoring.
            if not self.config.allow_scene_strong_pairs:
                for (region_a, region_b), score in list(pair_scores.items()):
                    type_a = region_types.get(region_a, "unknown")
                    type_b = region_types.get(region_b, "unknown")
                    if (type_a == "scene" or type_b == "scene") and score > 0.0:
                        pair_scores[(region_a, region_b)] = 0.0

        strong_pairs: List[Tuple[int, int, float]] = []
        weak_pairs: List[Tuple[int, int, float]] = []
        for (region_a, region_b), score in pair_scores.items():
            if score >= self.config.strong_threshold:
                strong_pairs.append((region_a, region_b, score))
            elif score <= self.config.weak_threshold:
                weak_pairs.append((region_a, region_b, score))

        strong_pairs = sorted(strong_pairs, key=lambda x: x[2], reverse=True)[: self.config.max_pairs_to_adjust]
        weak_pairs = sorted(weak_pairs, key=lambda x: x[2])[: self.config.max_pairs_to_adjust]
        score_map = {f"{a}-{b}": s for (a, b), s in pair_scores.items()}
        return strong_pairs, weak_pairs, score_map

    def _count_entity_mentions(self, text: str, entity_aliases: Dict[int, List[str]]) -> int:
        normalized = _normalize_text(text)
        count = 0
        for aliases in entity_aliases.values():
            if self._mention_entity(normalized, aliases):
                count += 1
        return count

    def _infer_region_types(self, inputs: dict, processed_inputs: dict) -> Dict[int, str]:
        ref_num = int(processed_inputs["ref_num"])
        region_start = int(processed_inputs["latents_start_index"])
        region_types: Dict[int, str] = {}

        for i in range(1, ref_num + 1):
            region_idx = region_start + i - 1
            ref_info = inputs.get(f"ref_img_{i}", {})
            image_path = str(ref_info.get("image_path", "") or "").lower()
            if "/scene/" in image_path:
                region_types[region_idx] = "scene"
            elif "/human/" in image_path:
                region_types[region_idx] = "human"
            elif "/animal/" in image_path:
                region_types[region_idx] = "animal"
            elif "/clothes/" in image_path:
                region_types[region_idx] = "clothes"
            elif "/object/" in image_path:
                region_types[region_idx] = "object"
            else:
                region_types[region_idx] = "unknown"

        # If there is an uncontrolled region token group, mark it explicitly.
        region_indices = list(processed_inputs.get("region_index_list", []))
        if len(region_indices) > ref_num:
            for ridx in region_indices[ref_num:]:
                region_types[int(ridx)] = "uncontrolled"

        return region_types

    def analyze(self, inputs: dict, processed_inputs: dict) -> Dict[str, object]:
        if not self.config.enabled:
            self.last_report = {
                "enabled": False,
                "status": "disabled",
                "strong_pairs": [],
                "weak_pairs": [],
                "pair_scores": {},
            }
            return self.last_report

        entity_aliases = self._build_entity_aliases(inputs, processed_inputs)
        region_types = self._infer_region_types(inputs, processed_inputs)
        context_text = self._build_context_text(inputs, processed_inputs)
        strong_pairs, weak_pairs, pair_scores = self._extract_pairs(
            context_text=context_text,
            entity_aliases=entity_aliases,
            region_types=region_types,
        )
        used_sad_desc_fallback = False
        used_prompt_fallback = False

        cei_text = str(inputs.get("CEI", "") or "").strip()
        prompt_text = str(inputs.get("prompt", "") or "").strip()
        if (
            self.config.fallback_to_prompt_when_cei_has_no_entity
            and cei_text
            and prompt_text
            and len(strong_pairs) == 0
            and len(weak_pairs) == 0
        ):
            mention_count = self._count_entity_mentions(cei_text, entity_aliases)
            if mention_count < 2:
                merged_context = f"{cei_text}. {prompt_text}".strip()
                merged_strong, merged_weak, merged_scores = self._extract_pairs(
                    context_text=merged_context,
                    entity_aliases=entity_aliases,
                    region_types=region_types,
                )
                if len(merged_strong) > 0 or len(merged_weak) > 0:
                    context_text = merged_context
                    strong_pairs = merged_strong
                    weak_pairs = merged_weak
                    pair_scores = merged_scores
                    used_prompt_fallback = True

        if (
            self.config.fallback_to_sad_desc_when_no_pairs
            and len(strong_pairs) == 0
            and len(weak_pairs) == 0
        ):
            fallback_context = self._build_context_text(inputs, processed_inputs, force_include_sad_desc=True)
            fallback_strong, fallback_weak, fallback_scores = self._extract_pairs(
                context_text=fallback_context,
                entity_aliases=entity_aliases,
                region_types=region_types,
            )
            if len(fallback_strong) > 0 or len(fallback_weak) > 0:
                context_text = fallback_context
                strong_pairs = fallback_strong
                weak_pairs = fallback_weak
                pair_scores = fallback_scores
                used_sad_desc_fallback = True

        used_single_pair_fallback = False
        if (
            self.config.assume_single_pair_when_cei_relation_only
            and len(strong_pairs) == 0
            and len(weak_pairs) == 0
        ):
            region_ids = sorted(entity_aliases.keys())
            if len(region_ids) == 2:
                cei_score = self._score_clause(_normalize_text(cei_text))
                if cei_score >= self.config.strong_threshold:
                    strong_pairs = [(int(region_ids[0]), int(region_ids[1]), float(cei_score))]
                    pair_scores = {f"{region_ids[0]}-{region_ids[1]}": float(cei_score)}
                    used_single_pair_fallback = True
                elif cei_score <= self.config.weak_threshold:
                    weak_pairs = [(int(region_ids[0]), int(region_ids[1]), float(cei_score))]
                    pair_scores = {f"{region_ids[0]}-{region_ids[1]}": float(cei_score)}
                    used_single_pair_fallback = True

        self.last_report = {
            "enabled": True,
            "status": "ok" if context_text else "no_text",
            "context_text": context_text,
            "strong_pairs": strong_pairs,
            "weak_pairs": weak_pairs,
            "pair_scores": pair_scores,
            "used_sad_desc_fallback": used_sad_desc_fallback,
            "used_prompt_fallback": used_prompt_fallback,
            "used_single_pair_fallback": used_single_pair_fallback,
            "region_types": {str(k): v for k, v in region_types.items()},
        }
        return self.last_report

    @staticmethod
    def _effective_region_tokens(processed_inputs: dict, region_idx: int) -> torch.Tensor:
        if region_idx in processed_inputs.get("mitigated_indices", {}):
            return processed_inputs["mitigated_indices"][region_idx]
        return processed_inputs["token_indices"][region_idx]

    @staticmethod
    def _is_additive_bias_mask(mask: torch.Tensor) -> bool:
        return bool(mask.dtype.is_floating_point and float(mask.min().item()) < -1e-4)

    @classmethod
    def _set_pair_block(cls, mask: torch.Tensor, row_tokens: torch.Tensor, col_tokens: torch.Tensor, value: bool) -> None:
        if row_tokens.numel() == 0 or col_tokens.numel() == 0:
            return
        rows, cols = torch.meshgrid(row_tokens, col_tokens, indexing="ij")
        if mask.dtype == torch.bool:
            mask[rows, cols] = bool(value)
            return

        if cls._is_additive_bias_mask(mask):
            # Additive bias convention: open=0, blocked=negative.
            fill_value = 0.0 if value else float(mask.min().item())
            mask[rows, cols] = torch.tensor(fill_value, dtype=mask.dtype, device=mask.device)
            return

        # Fallback for non-bool, non-additive masks (e.g., 0/1 float masks).
        mask[rows, cols] = float(value)

    @classmethod
    def _pair_open_ratio(cls, mask: torch.Tensor, row_tokens: torch.Tensor, col_tokens: torch.Tensor) -> float:
        if row_tokens.numel() == 0 or col_tokens.numel() == 0:
            return 0.0
        rows, cols = torch.meshgrid(row_tokens, col_tokens, indexing="ij")
        pair_values = mask[rows, cols]
        if mask.dtype == torch.bool:
            return float(pair_values.to(torch.float32).mean().item())

        pair_values = pair_values.to(torch.float32)
        if cls._is_additive_bias_mask(mask):
            # Normalize additive bias to [0,1] openness for readable diagnostics.
            blocked_value = float(mask.min().item())
            if blocked_value < -1e-8:
                normalized = (pair_values - blocked_value) / (-blocked_value)
                normalized = normalized.clamp_(0.0, 1.0)
                return float(normalized.mean().item())
            return 1.0

        # Fallback: interpret non-negative values as open.
        return float((pair_values >= 0.5).to(torch.float32).mean().item())

    def apply(
        self,
        attention_mask: torch.Tensor,
        processed_inputs: dict,
        binding_report: Dict[str, object],
        stage: int,
    ) -> torch.Tensor:
        if not self.config.enabled:
            return attention_mask
        if not binding_report or binding_report.get("enabled") is not True:
            return attention_mask

        apply_strong = self.config.apply_stage1_strong if stage == 1 else self.config.apply_stage2_strong
        apply_weak = self.config.apply_stage1_weak if stage == 1 else self.config.apply_stage2_weak
        suppress_non_strong = self.config.suppress_non_strong_stage1 if stage == 1 else self.config.suppress_non_strong_stage2
        strong_pairs = list(binding_report.get("strong_pairs", []))
        weak_pairs = list(binding_report.get("weak_pairs", []))

        # Safety guard: weak-only evidence is noisy in practice and can over-prune
        # cross-region links. Keep CEI conservative unless we have at least one
        # strong relation anchor in text.
        if self.config.disable_weak_when_no_strong and len(strong_pairs) == 0:
            apply_weak = False

        if self.config.require_strong_pairs_for_suppression and len(strong_pairs) == 0:
            suppress_non_strong = False

        if binding_report.get("used_single_pair_fallback", False):
            if stage == 1 and not self.config.single_pair_fallback_apply_stage1:
                apply_strong = False
            if stage == 2 and not self.config.single_pair_fallback_apply_stage2:
                apply_strong = False

        if not apply_strong and not apply_weak and not suppress_non_strong:
            return attention_mask

        out = attention_mask.clone()
        strong_set = {
            tuple(sorted((int(region_a), int(region_b))))
            for region_a, region_b, _ in strong_pairs
        }
        weak_set = {
            tuple(sorted((int(region_a), int(region_b))))
            for region_a, region_b, _ in weak_pairs
        }

        strong_before_after = []
        weak_before_after = []
        suppressed_pairs = []

        if apply_strong:
            for region_a, region_b, _ in strong_pairs:
                tokens_a = self._effective_region_tokens(processed_inputs, int(region_a))
                tokens_b = self._effective_region_tokens(processed_inputs, int(region_b))
                before_ratio = self._pair_open_ratio(out, tokens_a, tokens_b)
                self._set_pair_block(out, tokens_a, tokens_b, True)
                self._set_pair_block(out, tokens_b, tokens_a, True)
                after_ratio = self._pair_open_ratio(out, tokens_a, tokens_b)
                strong_before_after.append((int(region_a), int(region_b), before_ratio, after_ratio))

        if apply_weak:
            for region_a, region_b, _ in weak_pairs:
                tokens_a = self._effective_region_tokens(processed_inputs, int(region_a))
                tokens_b = self._effective_region_tokens(processed_inputs, int(region_b))
                before_ratio = self._pair_open_ratio(out, tokens_a, tokens_b)
                self._set_pair_block(out, tokens_a, tokens_b, False)
                self._set_pair_block(out, tokens_b, tokens_a, False)
                after_ratio = self._pair_open_ratio(out, tokens_a, tokens_b)
                weak_before_after.append((int(region_a), int(region_b), before_ratio, after_ratio))

        if (
            suppress_non_strong
            and self.config.disable_suppress_when_no_pairs
            and len(strong_pairs) == 0
            and len(weak_pairs) == 0
        ):
            suppress_non_strong = False

        if suppress_non_strong:
            region_index_list = list(processed_inputs.get("region_index_list", []))
            region_types_raw = binding_report.get("region_types", {})
            region_types: Dict[int, str] = {
                int(k): str(v) for k, v in region_types_raw.items()
            } if isinstance(region_types_raw, dict) else {}

            human_regions = {
                rid for rid, rtype in region_types.items() if rtype == "human"
            }

            def _has_human_strong_anchor(region_id: int) -> bool:
                for hr in human_regions:
                    if tuple(sorted((int(region_id), int(hr)))) in strong_set:
                        return True
                return False

            for idx_a in range(len(region_index_list)):
                for idx_b in range(idx_a + 1, len(region_index_list)):
                    region_a = int(region_index_list[idx_a])
                    region_b = int(region_index_list[idx_b])
                    pair_key = tuple(sorted((region_a, region_b)))
                    if pair_key in strong_set:
                        continue
                    if pair_key in weak_set:
                        continue

                    type_a = region_types.get(region_a, "unknown")
                    type_b = region_types.get(region_b, "unknown")

                    if self.config.preserve_uncontrolled_region_links and (
                        type_a == "uncontrolled" or type_b == "uncontrolled"
                    ):
                        continue

                    if self.config.suppress_scene_pairs_only and not (
                        type_a == "scene" or type_b == "scene"
                    ):
                        continue

                    if self.config.preserve_human_scene_links:
                        if type_a == "scene" and type_b in ("human", "animal"):
                            continue
                        if type_b == "scene" and type_a in ("human", "animal"):
                            continue

                    if self.config.preserve_human_linked_scene_links:
                        if type_a == "scene" and _has_human_strong_anchor(region_b):
                            continue
                        if type_b == "scene" and _has_human_strong_anchor(region_a):
                            continue

                    tokens_a = self._effective_region_tokens(processed_inputs, region_a)
                    tokens_b = self._effective_region_tokens(processed_inputs, region_b)
                    before_ratio = self._pair_open_ratio(out, tokens_a, tokens_b)
                    self._set_pair_block(out, tokens_a, tokens_b, False)
                    self._set_pair_block(out, tokens_b, tokens_a, False)
                    after_ratio = self._pair_open_ratio(out, tokens_a, tokens_b)
                    suppressed_pairs.append((region_a, region_b, before_ratio, after_ratio))

        self.last_apply_stats = {
            "stage": int(stage),
            "apply_strong": bool(apply_strong),
            "apply_weak": bool(apply_weak),
            "strong_before_after": strong_before_after,
            "weak_before_after": weak_before_after,
            "suppressed_pairs": suppressed_pairs,
            "suppressed_non_strong": bool(suppress_non_strong),
        }

        return out
