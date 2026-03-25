import torch
from codes import FluxLamicPipeline, LamicProcessor
from codes.transformer_flux_lamic import FluxLamicTransformer2DModel
from codes.soft_rma import SoftRMAProcessor
from codes.cei_binding import CEIBindingProcessor
import json
import os
import torchvision
from torchao.quantization import quantize_, int8_weight_only
from tqdm import tqdm
from utils.image_utils import image_grid
from utils.visualize_bbox_mask import overlay_bbox_masks, overlay_bbox_masks_advanced

def main(args):
    torch_dtype = torch.bfloat16

    flux_kontext_transformer_path = args.flux_kontext_transformer_path
    flux_path = args.flux_path
    transformer = FluxLamicTransformer2DModel.from_pretrained(flux_kontext_transformer_path)
    pipe = FluxLamicPipeline.from_pretrained(flux_path,
                                             transformer=transformer,
                                             torch_dtype=torch_dtype)
    
    print("Applying INT8 weight-only quantization to Transformer and Text Encoder...")
    quantize_(pipe.transformer, int8_weight_only())  # Transformer -> INT8 weights
    if hasattr(pipe, "text_encoder_2"):
        quantize_(pipe.text_encoder_2, int8_weight_only())  # Text Encoder -> INT8 weights

    pipe = pipe.to(torch_dtype)
    if args.reduce_memory_usage:
        # Keep model offloaded to reduce peak VRAM during load/inference.
        pipe.enable_model_cpu_offload()
        pipe.enable_attention_slicing()
        pipe.enable_vae_slicing()
    else:
        pipe.to("cuda")

    save_folder = args.save_folder
    input_path = args.input_path
    os.makedirs(save_folder, exist_ok=True)
    os.makedirs(os.path.join(save_folder, "masks"), exist_ok=True)
    if args.save_bbox_masks:
        os.makedirs(os.path.join(save_folder, "bboxed"), exist_ok=True)

    all_inputs = json.load(open(input_path))
    num_samples = len(all_inputs)
    num_img_per_sample = args.num_img_per_sample
    
    init_seed = args.init_seed
    output_height = args.output_height
    output_width = args.output_width
    first_stage_ratio = args.first_stage_ratio
    second_stage_ratio = args.second_stage_ratio
    logic_map_path = args.logic_map_path
    auto_resize_ref_img = args.auto_resize_ref_img
    fix_ref_img_size = args.fix_ref_img_size
    ref_size_height = args.ref_size_height
    ref_size_width = args.ref_size_width
    resize_output_size_in_advance = args.resize_output_size_in_advance
    dynamic_mask_config = None
    if args.enable_dynamic_mask_evolution:
        dynamic_mask_config = json.load(open(args.dynamic_mask_config_path))
        dynamic_mask_config["enabled"] = True

    soft_rma_processor = None
    if args.enable_soft_rma:
        soft_rma_config = json.load(open(args.soft_rma_config_path))
        soft_rma_config["enabled"] = True
        soft_rma_processor = SoftRMAProcessor(config=soft_rma_config)

    cei_binding_processor = None
    if args.enable_cei_binding:
        cei_binding_config = json.load(open(args.cei_binding_config_path))
        cei_binding_config["enabled"] = True
        cei_binding_processor = CEIBindingProcessor(config=cei_binding_config)

    lamic_processor = LamicProcessor(logic_map_path, auto_resize_ref_img, fix_ref_img_size, ref_size_height, ref_size_width)
    generator = torch.Generator(device="cuda")

    test_bar = tqdm(range(num_samples), desc="Processing samples")
    for i in range(num_samples):
        if args.choose_sample is not None:
            if i not in args.choose_sample:
                continue
        if i < args.start_sample:
            continue
        if args.concat_per_sample:
            image_list = []
            bboxed_image_list = []

        inputs = all_inputs[f"sample_{i:03d}"]
        processed_inputs = lamic_processor(inputs, pipe, height=output_height, width=output_width, resize_output_size=resize_output_size_in_advance,
                                            padding_prompts=True, max_sequence_length=512, save_bbox_masks=args.save_bbox_masks)
        cei_binding_report = None
        if cei_binding_processor is not None:
            cei_binding_report = cei_binding_processor.analyze(inputs=inputs, processed_inputs=processed_inputs)
        # get index mask for two stages
        index_mask = lamic_processor._get_index_mask(inputs=inputs, processed_inputs=processed_inputs)
        # get attention mask for two stages (if there is mitigate_overlapping_region, attention mask 2 must be first generated, because it will not be mitigated)
        attention_mask_2 = lamic_processor._get_attention_mask(index_mask=index_mask[:, :, 1], processed_inputs=processed_inputs, stage=2)
        print("indices before mitigate_overlapping_region: ", [len(token_indice) for token_indice in processed_inputs["token_indices"].values()])
        attention_mask_1_before_mitigate = lamic_processor._get_attention_mask(index_mask=index_mask[:, :, 0], processed_inputs=processed_inputs, stage=1)
        # mitigate overlapping region
        processed_inputs = lamic_processor._mitigate_overlapping_region(inputs=inputs, processed_inputs=processed_inputs)
        print("indices after mitigate_overlapping_region: ", [index for index in processed_inputs['mitigated_indices'].keys()], [len(region_indices) for region_indices in processed_inputs['mitigated_indices'].values()])
        attention_mask_1 = lamic_processor._get_attention_mask(index_mask=index_mask[:, :, 0], processed_inputs=processed_inputs, stage=1)

        if soft_rma_processor is not None:
            attention_mask_1 = soft_rma_processor.apply(
                hard_mask=attention_mask_1,
                processed_inputs=processed_inputs,
                stage=1,
            )
            stats_stage1 = dict(soft_rma_processor.last_stats)
            attention_mask_2 = soft_rma_processor.apply(
                hard_mask=attention_mask_2,
                processed_inputs=processed_inputs,
                stage=2,
            )
            stats_stage2 = dict(soft_rma_processor.last_stats)
            print(
                "Soft-RMA applied. "
                f"stage1_softened_ratio={stats_stage1.get('softened_ratio', 0.0):.6f}, "
                f"stage1_scale={stats_stage1.get('stage_scale', 1.0):.2f}, "
                f"stage2_softened_ratio={stats_stage2.get('softened_ratio', 0.0):.6f}, "
                f"stage2_scale={stats_stage2.get('stage_scale', 1.0):.2f}, "
                f"stage2_ref_softened_pairs={int(stats_stage2.get('ref_softened_pairs', 0.0))}, "
                f"stage2_prompt_softened_pairs={int(stats_stage2.get('prompt_softened_pairs', 0.0))}, "
                f"boundary_token_ratio={stats_stage1.get('boundary_token_ratio', 0.0):.6f}"
            )

        if soft_rma_processor is None:
            print("difference between attention mask 1 before and after mitigate_overlapping_region: ", torch.mean(torch.abs(attention_mask_1_before_mitigate.to(torch.float32) - attention_mask_1.to(torch.float32))))
        else:
            hard_density = float(attention_mask_1_before_mitigate.to(torch.float32).mean().item())
            soft_bias_mean = float(attention_mask_1.to(torch.float32).mean().item())
            print(f"Soft-RMA mask stats: hard_density={hard_density:.6f}, soft_bias_mean={soft_bias_mean:.6f}")

        if cei_binding_processor is not None and cei_binding_report is not None:
            attention_mask_1_before_cei = attention_mask_1.clone()
            attention_mask_2_before_cei = attention_mask_2.clone()
            attention_mask_1 = cei_binding_processor.apply(
                attention_mask=attention_mask_1,
                processed_inputs=processed_inputs,
                binding_report=cei_binding_report,
                stage=1,
            )
            apply_stats_stage1 = dict(cei_binding_processor.last_apply_stats)
            attention_mask_2 = cei_binding_processor.apply(
                attention_mask=attention_mask_2,
                processed_inputs=processed_inputs,
                binding_report=cei_binding_report,
                stage=2,
            )
            apply_stats_stage2 = dict(cei_binding_processor.last_apply_stats)
            changed_ratio_stage1 = float((attention_mask_1_before_cei != attention_mask_1).to(torch.float32).mean().item())
            changed_ratio_stage2 = float((attention_mask_2_before_cei != attention_mask_2).to(torch.float32).mean().item())
            latents_start_index = int(processed_inputs["latents_start_index"])
            strong_before_after_stage1_local = [
                (
                    int(a) - latents_start_index + 1,
                    int(b) - latents_start_index + 1,
                    float(before),
                    float(after),
                )
                for a, b, before, after in apply_stats_stage1.get("strong_before_after", [])
            ]
            weak_before_after_stage1_local = [
                (
                    int(a) - latents_start_index + 1,
                    int(b) - latents_start_index + 1,
                    float(before),
                    float(after),
                )
                for a, b, before, after in apply_stats_stage1.get("weak_before_after", [])
            ]
            suppressed_pairs_stage1_local = [
                (
                    int(a) - latents_start_index + 1,
                    int(b) - latents_start_index + 1,
                    float(before),
                    float(after),
                )
                for a, b, before, after in apply_stats_stage1.get("suppressed_pairs", [])
            ]
            strong_before_after_stage2_local = [
                (
                    int(a) - latents_start_index + 1,
                    int(b) - latents_start_index + 1,
                    float(before),
                    float(after),
                )
                for a, b, before, after in apply_stats_stage2.get("strong_before_after", [])
            ]
            weak_before_after_stage2_local = [
                (
                    int(a) - latents_start_index + 1,
                    int(b) - latents_start_index + 1,
                    float(before),
                    float(after),
                )
                for a, b, before, after in apply_stats_stage2.get("weak_before_after", [])
            ]
            suppressed_pairs_stage2_local = [
                (
                    int(a) - latents_start_index + 1,
                    int(b) - latents_start_index + 1,
                    float(before),
                    float(after),
                )
                for a, b, before, after in apply_stats_stage2.get("suppressed_pairs", [])
            ]
            strong_pairs_local = [
                (int(a) - latents_start_index + 1, int(b) - latents_start_index + 1, float(s))
                for a, b, s in cei_binding_report.get("strong_pairs", [])
            ]
            weak_pairs_local = [
                (int(a) - latents_start_index + 1, int(b) - latents_start_index + 1, float(s))
                for a, b, s in cei_binding_report.get("weak_pairs", [])
            ]
            print(
                "CEI binding applied. "
                f"status={cei_binding_report.get('status', 'na')}, "
                f"strong_pairs={len(cei_binding_report.get('strong_pairs', []))}, "
                f"weak_pairs={len(cei_binding_report.get('weak_pairs', []))}, "
                f"used_sad_desc_fallback={cei_binding_report.get('used_sad_desc_fallback', False)}, "
                f"used_prompt_fallback={cei_binding_report.get('used_prompt_fallback', False)}, "
                f"used_single_pair_fallback={cei_binding_report.get('used_single_pair_fallback', False)}, "
                f"stage1_changed_ratio={changed_ratio_stage1:.6f}, "
                f"stage2_changed_ratio={changed_ratio_stage2:.6f}, "
                f"strong_pairs_local={strong_pairs_local}, "
                f"weak_pairs_local={weak_pairs_local}, "
                f"stage1_suppressed_non_strong={apply_stats_stage1.get('suppressed_non_strong', False)}, "
                f"stage1_apply_strong={apply_stats_stage1.get('apply_strong', False)}, "
                f"stage1_strong_before_after_local={strong_before_after_stage1_local}, "
                f"stage1_weak_before_after_local={weak_before_after_stage1_local}, "
                f"stage1_suppressed_pairs_local={suppressed_pairs_stage1_local}, "
                f"stage2_suppressed_non_strong={apply_stats_stage2.get('suppressed_non_strong', False)}, "
                f"stage2_apply_strong={apply_stats_stage2.get('apply_strong', False)}, "
                f"stage2_strong_before_after_local={strong_before_after_stage2_local}, "
                f"stage2_weak_before_after_local={weak_before_after_stage2_local}, "
                f"stage2_suppressed_pairs_local={suppressed_pairs_stage2_local}, "
                f"context_text={str(cei_binding_report.get('context_text', ''))[:180]}, "
                f"region_types={cei_binding_report.get('region_types', {})}, "
                f"pair_scores_top5={sorted(cei_binding_report.get('pair_scores', {}).items(), key=lambda kv: kv[1], reverse=True)[:5]}"
            )

        save_masks(index_mask, attention_mask_1, attention_mask_2, save_folder=os.path.join(save_folder, "masks"))

        # full attention
        if args.full_attention:
            attention_mask_2 = torch.ones_like(attention_mask_2).to(torch.bool)
            attention_mask_1 = torch.ones_like(attention_mask_1).to(torch.bool)

        # Generate images for each sample
        for j in range(num_img_per_sample):
            generator.manual_seed(init_seed + 2 * j)
            test_bar.set_description(f"Processing sample {i:03d} image {j + 1}")
            image = pipe(
                image=processed_inputs["images"], # list of images
                prompt=processed_inputs.get("clip_prompt", processed_inputs["total_prompts"]),
                prompt_2=processed_inputs["total_prompts"],
                attention_mask1=attention_mask_1,
                attention_mask2=attention_mask_2,
                guidance_scale=args.guidance_scale,
                num_inference_steps=args.num_inference_steps,
                height=processed_inputs["output_height"],
                width=processed_inputs["output_width"],
                generator=generator,
                _auto_resize=auto_resize_ref_img,
                height_width_is_adjusted=True,
                first_stage_ratio=inputs["first_stage_ratio"] if "first_stage_ratio" in inputs else first_stage_ratio,
                second_stage_ratio=inputs["second_stage_ratio"] if "second_stage_ratio" in inputs else second_stage_ratio,
                enhance_factors=processed_inputs["enhance_factors"],
                dynamic_mask_evolution=dynamic_mask_config,
                dynamic_mask_inputs={
                    "lamic_processor": lamic_processor,
                    "processed_inputs": processed_inputs,
                    "index_mask_stage2": index_mask[:, :, 1],
                    "inputs": inputs,
                } if dynamic_mask_config is not None else None,
            ).images[0]
            # save single image
            image_save_path = os.path.join(save_folder, f"sample_{i:03d}_image_{j +1}.png") 
            image.save(image_save_path)
            print(f"image 'sample_{i:03d}_image_{j + 1}.png' saved in {image_save_path}")
            
            if args.concat_per_sample:
                image_list.append(image)
            
            if isinstance(processed_inputs["bbox_masks_list"], list) and len(processed_inputs["bbox_masks_list"]) > 0:
                print(processed_inputs["bbox_masks_list"])
                bboxed_image = overlay_bbox_masks_advanced(image, processed_inputs["bbox_masks_list"], fill_alpha=128, outline_only=False)
                # save single bboxed image
                bboxed_image_save_path = os.path.join(save_folder, 'bboxed', f"sample_{i:03d}_image_{j + 1}_bboxed.png")
                bboxed_image.save(bboxed_image_save_path)
                print(f"bboxed image 'sample_{i:03d}_image_{j + 1}_bboxed.png' saved in {bboxed_image_save_path}")
                if args.concat_per_sample:
                    bboxed_image_list.append(bboxed_image)

        if args.concat_per_sample:
            rows = max(1, len(image_list) // 2)
            cols = max(1, min(2, len(image_list)))
            image = image_grid(image_list, rows, cols)
            save_path = os.path.join(save_folder, f"sample_{i:03d}.png")
            image.save(save_path)
            print(f"Save results sample_{i:03d} to: {save_path}")
            for j in range(num_img_per_sample):
                os.remove(os.path.join(save_folder, f"sample_{i:03d}_image_{j + 1}.png"))
            
            if len(bboxed_image_list) > 0:
                bbox_rows = max(1, len(bboxed_image_list) // 2)
                bbox_cols = max(1, min(2, len(bboxed_image_list)))
                bboxed_image = image_grid(bboxed_image_list, bbox_rows, bbox_cols)
                bboxed_save_path = os.path.join(save_folder, 'bboxed', f"sample_{i:03d}_bboxed.png")
                bboxed_image.save(bboxed_save_path)
                print(f"Save bboxed results sample_{i:03d} to: {bboxed_save_path}")
                for j in range(num_img_per_sample):
                    os.remove(os.path.join(save_folder, 'bboxed', f"sample_{i:03d}_image_{j + 1}_bboxed.png"))  
                    
            del image_list, bboxed_image_list


def save_masks(index_mask, attention_mask1, attention_mask2, save_folder):
    os.makedirs(save_folder, exist_ok=True)
    to_pil = torchvision.transforms.ToPILImage()
    pil_index_mask_stage1 = to_pil(index_mask[:, :, 0])
    pil_index_mask_stage2 = to_pil(index_mask[:, :, 1])
    pil_index_mask_fuse = to_pil(torch.mean(index_mask, dim=-1))
    pil_attention_mask1 = to_pil(attention_mask1.to(torch.float32))
    pil_attention_mask2 = to_pil(attention_mask2.to(torch.float32))
    pil_index_mask_stage1.save(os.path.join(save_folder, "index_mask_stage1.png"))
    pil_index_mask_stage2.save(os.path.join(save_folder, "index_mask_stage2.png"))
    pil_index_mask_fuse.save(os.path.join(save_folder, "index_mask_fuse.png"))
    pil_attention_mask1.save(os.path.join(save_folder, "attention_mask1.png"))
    pil_attention_mask2.save(os.path.join(save_folder, "attention_mask2.png"))

import argparse
def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_folder", type=str, default="./results/cmp")
    parser.add_argument("--input_path", type=str, default="./dataset/structured_inputs/Four-Reference.json")
    parser.add_argument("--first_stage_ratio", type=float, default=0.05)
    parser.add_argument("--num_img_per_sample", type=int, default=4)
    parser.add_argument("--concat_per_sample", type=bool, default=False)
    parser.add_argument("--init_seed", type=int, default=42)
    parser.add_argument("--output_height", type=int, default=1024)
    parser.add_argument("--output_width", type=int, default=1024)
    parser.add_argument("--guidance_scale", type=float, default=2.5)
    parser.add_argument("--num_inference_steps", type=int, default=20)
    parser.add_argument("--second_stage_ratio", type=float, default=1.0)
    parser.add_argument("--logic_map_path", type=str, default="configs/attention_mask_logic_map.json")
    parser.add_argument("--auto_resize_ref_img", type=bool, default=False)
    parser.add_argument("--fix_ref_img_size", type=bool, default=False)
    parser.add_argument("--ref_size_height", type=int, default=256)
    parser.add_argument("--ref_size_width", type=int, default=256)
    parser.add_argument("--resize_output_size_in_advance", type=bool, default=True)
    parser.add_argument("--save_bbox_masks", type=bool, default=True)
    parser.add_argument("--choose_sample", type=int, nargs="+", default=None)
    parser.add_argument("--start_sample", type=int, default=0)
    parser.add_argument("--full_attention", action="store_true")
    parser.add_argument("--flux_kontext_transformer_path", type=str, default="/mnt/sda/model_weights/FLUX.1-Kontext-dev/transformer/transformer",
                        help="path to the flux kontext transformer, diffuser format")
    parser.add_argument("--flux_path", type=str, default="/mnt/sata/models/FLUX.1-dev",
                        help="path to the flux model, diffuser format")
    parser.add_argument("--reduce_memory_usage", type=bool, default=True)
    parser.add_argument("--enable_dynamic_mask_evolution", action="store_true")
    parser.add_argument("--dynamic_mask_config_path", type=str, default="configs/dynamic_mask_evolution.json")
    parser.add_argument("--enable_soft_rma", action="store_true")
    parser.add_argument("--soft_rma_config_path", type=str, default="configs/soft_rma.json")
    parser.add_argument("--enable_cei_binding", action="store_true")
    parser.add_argument("--cei_binding_config_path", type=str, default="configs/cei_binding.json")
    return parser.parse_args()

if __name__ == "__main__":
    args = set_args()
    main(args)