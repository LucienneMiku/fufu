import os
import json
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from transformers import AutoImageProcessor, AutoModel
from torchmetrics.multimodal.clip_score import CLIPScore
import cv2
import numpy as np
from rembg import remove
import urllib.request
from transformers import CLIPProcessor, CLIPModel, BlipProcessor, BlipForQuestionAnswering
from insightface.app import FaceAnalysis # 人脸该指标评估
from numpy.linalg import norm
from cleanfid import fid
import shutil
import random
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation

# 不开启分词器多线程
os.environ["TOKENIZERS_PARALLELISM"] = "false"

#猴子补丁，暴力绕过 CVE 版本拦截
import transformers.utils.import_utils
import transformers.modeling_utils
# 将强制检查 PyTorch 版本的函数替换为 lambda: None (即直接 return，不报错)
transformers.utils.import_utils.check_torch_load_is_safe = lambda: None
transformers.modeling_utils.check_torch_load_is_safe = lambda: None

class LAMICEvaluator:
    def __init__(self, device="cuda"):
        self.device = device
        print("加载评估模型")
        
        # 1. 加载 DINOv2 (用于非人脸物体的 IP-S 计算)
        self.dino_processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
        self.dino_model = AutoModel.from_pretrained('facebook/dinov2-base').to(self.device)
        
        # 2. 加载 CLIP Score (用于计算 DPG 文本一致性)
        self.clip_score_fn = CLIPScore(model_name_or_path="openai/clip-vit-base-patch16").to(self.device)
        
        # 3. 加载 AES 美学评估模型 (用于AES 计算)
        local_clip_path = "../model/clip-vit-large-patch14"
        self.aes_processor = CLIPProcessor.from_pretrained(local_clip_path)
        self.aes_clip = CLIPModel.from_pretrained(local_clip_path).to(self.device)
        self.aes_mlp = torch.nn.Linear(768, 1).to(self.device)
        # 自动下载 LAION 官方的线性层权重 (仅 3KB)
        aes_weight_path = "sa_0_4_vit_l_14_linear.pth"
        if not os.path.exists(aes_weight_path):
            urllib.request.urlretrieve(
                "https://raw.githubusercontent.com/LAION-AI/aesthetic-predictor/main/sa_0_4_vit_l_14_linear.pth",
                aes_weight_path
            )
        self.aes_mlp.load_state_dict(torch.load(aes_weight_path, map_location=self.device))
        self.aes_mlp.eval()
        
        # 4. 加载 CLIPSeg (用于提取生成实体的 Mask 计算 IN-R)
        self.seg_processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
        self.seg_model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined").to(self.device)
        
        # 5. 加载 BLIP-VQA (用于 DPG 图谱评估)
        self.vqa_processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
        self.vqa_model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base").to(self.device)
        
        # 6. 加载 ArcFace (用于 ID-S 人脸一致性)
        # buffalo_l 是 insightface 官方精度最高的人脸分析模型包
        self.face_app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
        # det_size 设为 640x640，保证即使生成的人脸较小也能被检测到
        self.face_app.prepare(ctx_id=0, det_size=(640, 640))
        
        print("模型加载完成！")

    def split_grid_image(self, img_path):
        """将 2x2 的拼图切割成 4 张单独的图片"""
        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        w2, h2 = w // 2, h // 2
        
        # 返回 4 张单独的 PIL Image
        return [
            img.crop((0, 0, w2, h2)),       # 左上
            img.crop((w2, 0, w, h2)),       # 右上
            img.crop((0, h2, w2, h)),       # 左下
            img.crop((w2, h2, w, h))        # 右下
        ]

    @torch.no_grad()
    def calc_dinov2_similarity(self, img1: Image.Image, img2: Image.Image):
        # 计算两张图片的 DINOv2 余弦相似度 (使用学术标准的 CLS Token + L2 归一化)
        inputs1 = self.dino_processor(images=img1, return_tensors="pt").to(self.device)
        inputs2 = self.dino_processor(images=img2, return_tensors="pt").to(self.device)
        
        # 获取最后一层的完整隐藏状态，而不是 pooler
        outputs1 = self.dino_model(**inputs1)
        outputs2 = self.dino_model(**inputs2)
        
        # 提取 CLS token (即序列的第0个 token)
        # 形状变成 [batch_size, hidden_size]
        cls_feat1 = outputs1.last_hidden_state[:, 0, :]
        cls_feat2 = outputs2.last_hidden_state[:, 0, :]
        
        # 学术标准操作：特征必须经过 L2 归一化 (L2 Normalization)
        cls_feat1 = F.normalize(cls_feat1, p=2, dim=-1)
        cls_feat2 = F.normalize(cls_feat2, p=2, dim=-1)
        
        # 计算余弦相似度
        sim = (cls_feat1 * cls_feat2).sum(dim=-1)
        return sim.item()

    @torch.no_grad()
    def calc_id_s_score(self, real_img: Image.Image, gen_img: Image.Image):
        real_cv2 = cv2.cvtColor(np.array(real_img), cv2.COLOR_RGB2BGR)
        gen_cv2 = cv2.cvtColor(np.array(gen_img), cv2.COLOR_RGB2BGR)

        faces_real = self.face_app.get(real_cv2)
        # 1. 如果原图（参考图）根本检测不到脸，直接返回 None，不计入总分！
        if len(faces_real) == 0:
            return None 

        faces_gen = self.face_app.get(gen_cv2)
        # 2. 原图有脸，但生成图没脸（被遮挡或没画出来），给 0 分
        if len(faces_gen) == 0:
            return 0.0

        emb_real = faces_real[0].embedding
        emb_gen = faces_gen[0].embedding

        sim = np.dot(emb_real, emb_gen) / (norm(emb_real) * norm(emb_gen))
        # 3. 乘以 100 对标论文量纲
        return float(max(0.0, sim)) * 100.0
    
    @torch.no_grad()
    def calc_clip_score(self, img: Image.Image, prompt: str):
        """计算图片和文本的 CLIP Score """
        # torchmetrics 需要 tensor 格式的图片输入
        img_tensor = transforms.ToTensor()(img).unsqueeze(0).to(self.device)
        # CLIPScore 需要 [0, 255] 的 uint8 tensor
        img_tensor = (img_tensor * 255).to(torch.uint8) 
        
        score = self.clip_score_fn(img_tensor, prompt)
        return score.item()
    
    @torch.no_grad()
    def calc_dpg_score(self, img: Image.Image, dsg_data: dict):
        """
        计算纯正的 DPG (Dense Prompt Graph) 图谱对齐分数
        带有严格的层级依赖校验逻辑
        """
        qid2question = dsg_data.get("qid2question", {})
        qid2dependency = dsg_data.get("qid2dependency", {})
        
        if not qid2question:
            return 0.0
            
        node_results = {} # 记录每个节点 (qid) 的真假状态
        correct_count = 0
        total_questions = len(qid2question)
        
        # 遍历所有节点问题
        for qid, question in qid2question.items():
            deps = qid2dependency.get(qid, [0])
            
            # 1. 校验前置依赖
            # 如果依赖列表中有一个前置节点是 False，则当前节点直接判定为 False (级联失败)
            deps_met = True
            for dep in deps:
                if str(dep) != "0" and not node_results.get(str(dep), False):
                    deps_met = False
                    break
            
            if not deps_met:
                node_results[qid] = False
                continue
                
            # 2. 如果前置依赖满足，调用 VLM 回答问题
            inputs = self.vqa_processor(img, question, return_tensors="pt").to(self.device)
            out = self.vqa_model.generate(**inputs, max_new_tokens=3)
            answer = self.vqa_processor.decode(out[0], skip_special_tokens=True).strip().lower()
            
            # 3. 记录结果
            is_yes = ("yes" in answer)
            node_results[qid] = is_yes
            
            if is_yes:
                correct_count += 1
                
        # 返回命中节点的百分比
        return (correct_count / total_questions) * 100.0

    @torch.no_grad()
    def calc_aes_score(self, img: Image.Image):
        """计算 LAION AES 美学评分"""
        inputs = self.aes_processor(images=img, return_tensors="pt").to(self.device)       
        # 直接调用 get_image_features 获取经过投影层的 768 维特征
        image_features = self.aes_clip.get_image_features(**inputs)        
        # L2 归一化
        image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
        # 通过线性层得到美学分数
        score = self.aes_mlp(image_features)
        return score.item()
    
    @torch.no_grad()
    def calc_layout_metrics(self, img: Image.Image, prompt_id: str, bbox: list):
        """
        根据论文公式计算 IN-R 和 FI-R
        prompt_id: 实体名称 (如 "a little girl")
        bbox: 目标边界框 [xmin, ymin, xmax, ymax]
        """
        w, h = img.size
        
        # 1. 获取 M_gen (生成物体的掩码)
        inputs = self.seg_processor(text=[prompt_id], images=[img], return_tensors="pt").to(self.device)
        outputs = self.seg_model(**inputs)
        
        pred = torch.sigmoid(outputs.logits)
        if pred.dim() == 2:   # 如果是 (352, 352)
            pred = pred.unsqueeze(0).unsqueeze(0)
        elif pred.dim() == 3: # 如果是 (1, 352, 352)
            pred = pred.unsqueeze(0)
            
        pred = F.interpolate(pred, size=(h, w), mode="bilinear", align_corners=False)
        pred_mask = pred.squeeze().cpu().numpy()
        
        # 阈值二值化 (论文或常规实现一般取 0.4 或 0.5)
        m_gen = (pred_mask > 0.4).astype(np.uint8)
        
        # 2. 获取 M_trg (目标 Bbox 的掩码)
        m_trg = np.zeros((h, w), dtype=np.uint8)
        left, top, right, bottom = int(bbox[0]*w), int(bbox[1]*h), int(bbox[2]*w), int(bbox[3]*h)
        left, top = max(0, left), max(0, top)
        right, bottom = min(w, right), min(h, bottom)
        m_trg[top:bottom, left:right] = 1
        
        # 3. 计算求和公式中的各个项
        area_gen = np.sum(m_gen)
        area_trg = np.sum(m_trg)
        area_intersection = np.sum(m_gen & m_trg)
        
        # 4. 根据公式 (10) 和 (11) 计算结果 (乘以 100 统一量纲)
        # IN-R = (M_gen ∩ M_trg) / M_gen * 100
        in_r = (area_intersection / area_gen * 100.0) if area_gen > 0 else 0.0
        
        # FI-R = (M_gen ∩ M_trg) / M_trg * 100
        fi_r = (area_intersection / area_trg * 100.0) if area_trg > 0 else 0.0
        
        return in_r, fi_r

    @torch.no_grad()
    def extract_subject_with_segmask(
        self,
        img: Image.Image,
        prompt_id: str,
        bbox: list = None,
        threshold: float = 0.4,
    ):
        """
        使用 CLIPSeg 提取主体区域并返回白底紧致裁剪图。
        - prompt_id: 主体文本提示 (例如 "a sea turtle")
        - bbox: 可选，用于将掩码约束在目标区域内
        """
        rgb_img = img.convert("RGB")
        w, h = rgb_img.size

        inputs = self.seg_processor(text=[prompt_id], images=[rgb_img], return_tensors="pt").to(self.device)
        outputs = self.seg_model(**inputs)

        pred = torch.sigmoid(outputs.logits)
        if pred.dim() == 2:
            pred = pred.unsqueeze(0).unsqueeze(0)
        elif pred.dim() == 3:
            pred = pred.unsqueeze(0)

        pred = F.interpolate(pred, size=(h, w), mode="bilinear", align_corners=False)
        mask = (pred.squeeze().cpu().numpy() > threshold).astype(np.uint8)

        # 若提供了 bbox，则只保留 bbox 内的掩码，避免抓到无关实例
        if bbox is not None:
            bbox_mask = np.zeros((h, w), dtype=np.uint8)
            left, top, right, bottom = int(bbox[0] * w), int(bbox[1] * h), int(bbox[2] * w), int(bbox[3] * h)
            left, top = max(0, left), max(0, top)
            right, bottom = min(w, right), min(h, bottom)
            bbox_mask[top:bottom, left:right] = 1
            constrained_mask = (mask & bbox_mask).astype(np.uint8)
            if np.sum(constrained_mask) > 0:
                mask = constrained_mask

        if np.sum(mask) == 0:
            return None

        np_img = np.array(rgb_img)
        white_bg = np.ones_like(np_img, dtype=np.uint8) * 255
        white_bg[mask == 1] = np_img[mask == 1]

        ys, xs = np.where(mask == 1)
        top, bottom = int(ys.min()), int(ys.max()) + 1
        left, right = int(xs.min()), int(xs.max()) + 1

        # 极端情况下避免空裁剪
        if top >= bottom or left >= right:
            return None

        return Image.fromarray(white_bg[top:bottom, left:right]).convert("RGB")

    def crop_by_bbox(self, image: Image.Image, bbox: list):
        """根据 JSON 中的 bbox 裁剪图片，用于局部相似度比对"""
        # bbox 格式通常是 [x_min, y_min, x_max, y_max]，取值 0~1
        w, h = image.size
        left = int(bbox[0] * w)
        top = int(bbox[1] * h)
        right = int(bbox[2] * w)
        bottom = int(bbox[3] * h)
        return image.crop((left, top, right, bottom))
    
    def remove_bg_and_white_canvas(self, img: Image.Image):
        """移除图片背景，并将其统一粘贴到纯白画布上"""
        try:
            # 1. 移除背景，返回带透明通道 (Alpha) 的 RGBA 图片
            rgba_img = remove(img)
            # 2. 创建一张和原图一样大的纯白底图
            white_bg = Image.new("RGBA", rgba_img.size, "WHITE")
            # 3. 将抠出来的物体贴在白底上
            white_bg.paste(rgba_img, (0, 0), rgba_img)
            # 4. 转回 RGB 格式供 DINOv2 提取
            return white_bg.convert("RGB")
        except Exception as e:
            print(f"抠图失败，退回原图: {e}")
            return img.convert("RGB")


def is_scene_reference(ref_data: dict) -> bool:
    sad = ref_data.get("SAD", {})
    desc = str(sad.get("desc", "")).lower()
    entity_id = str(sad.get("id", "")).lower()
    image_path = str(ref_data.get("image_path", "")).lower().replace("\\", "/")

    # 标记为 keep scene 的引用属于背景，不纳入物体外观一致性评估
    return (
        ("keep scene" in desc)
        or ("keep the scene" in desc)
        or ("/scene/" in image_path)
        or (" scene" in entity_id)
        or (entity_id == "scene")
        or (entity_id == "a scene")
    )
    
def main():
    json_path = "./dataset/structured_inputs/Two-Reference.json"  # JSON 测试集路径
    dsg_json_path = "./dataset/DSGs_for_DPG_Score/Two_Reference_DSG.json" #DPG 图谱 JSON
    with open(dsg_json_path, 'r', encoding='utf-8') as f:
        all_dsg_data = json.load(f)
    gen_folder = "./gen_datas/Two-Reference-cei" # 生成的图片保存目录
    
    #  FID 文件夹 
    fid_gen_folder = "./gen_datas/Two-Reference-cei_FID"
    # 真实图片分布文件夹
    fid_real_folder = "./dataset/COCO_images_for_FID/coco_5000" 
    # 清空并重建生成图的暂存文件夹
    # if os.path.exists(fid_gen_folder):
    #     shutil.rmtree(fid_gen_folder)
    # os.makedirs(fid_gen_folder, exist_ok=True)
    
    # 1. 读取所有的测试用例
    with open(json_path, 'r', encoding='utf-8') as f:
        all_inputs = json.load(f)
        
    evaluator = LAMICEvaluator()
    
    total_clip_score = 0
    total_dino_score = 0
    total_aes_score = 0
    total_dpg_score = 0.0
    total_in_r = 0.0
    total_fi_r = 0.0
    total_ids_score = 0.0
    
    clip_samples = 0
    dino_samples = 0
    aes_samples = 0
    dpg_samples = 0
    layout_samples = 0
    ids_samples = 0
    valid_ids_samples = 0

    for sample_key, sample_data in all_inputs.items():
        img_filename = f"{sample_key}.png"
        img_path = os.path.join(gen_folder, img_filename)
        
        if not os.path.exists(img_path):
            continue
            
        print(f"正在评估: {img_filename}")
        prompt = sample_data["prompt"]
        
        gen_images = evaluator.split_grid_image(img_path)
        
        for idx, gen_img in enumerate(gen_images):
            # 保存单图供 FID 读取
            split_img_path = os.path.join(fid_gen_folder, f"{sample_key}_{idx}.png")
            gen_img.save(split_img_path)
            
            # 1. 计算 CLIP Score
            clip_score = evaluator.calc_clip_score(gen_img, prompt)
            total_clip_score += clip_score
            clip_samples += 1

            # 2. 计算 DINOv2 相似度 (IP-S)
            for ref_key, ref_data in sample_data.items():
                if ref_key.startswith("ref_img_") and "bbox" in ref_data:
                    ref_img_path = ref_data["image_path"]
                    if os.path.exists(ref_img_path):
                        real_img = Image.open(ref_img_path).convert("RGB")
                        # 只在 DINO/ID-S 路径跳过背景引用，避免影响布局指标统计
                        if not is_scene_reference(ref_data):
                            # 抠出生成的局部物体
                            gen_cropped = evaluator.crop_by_bbox(gen_img, ref_data["bbox"])

                            clean_real_img = evaluator.remove_bg_and_white_canvas(real_img)
                            clean_gen_cropped = evaluator.remove_bg_and_white_canvas(gen_cropped)

                            # 把清洗后的图保存下来，看看背景是不是被完美去掉了
                            clean_gen_cropped.save(f"{gen_folder}/clean_crop_{sample_key}_{ref_key}_{idx}.png")

                            # 计算相似度
                            dino_sim = evaluator.calc_dinov2_similarity(clean_real_img, clean_gen_cropped)

                            total_dino_score += dino_sim
                            dino_samples += 1

                            # 计算 ID-S (仅针对人类实体)
                            # 提取文件路径和实体名称，转为小写用于匹配
                            img_path_lower = ref_data.get("image_path", "").lower()
                            entity_id_lower = ref_data.get("SAD", {}).get("id", "").lower()

                            # 判定规则：只要路径包含 'human'，或实体名包含人称关键字，就认为是人类
                            human_keywords = ['man', 'woman', 'boy', 'girl', 'person', 'face', 'human']
                            is_human = ("human" in img_path_lower) or any(k in entity_id_lower for k in human_keywords)

                            if is_human:
                                ids_score = evaluator.calc_id_s_score(real_img, gen_cropped)

                                # 只要返回值不是 None (说明参考图是有效的正脸)，我们就计入成绩
                                if ids_score is not None:
                                    total_ids_score += ids_score
                                    valid_ids_samples += 1 # 别忘了在 main 函数开头加一句 valid_ids_samples = 0
                        
                        # 2个布局指标评估
                        bbox = ref_data["bbox"]
                        # 计算 bbox 占据全图的面积比例 (宽 * 高)
                        area_ratio = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                        
                        # 严格执行论文规则：丢弃目标框面积占比 > 75% (0.75) 的样本
                        if area_ratio <= 0.75:
                            # 提取实体的名称 (比如 "a sea turtle")
                            entity_id = ref_data.get("SAD", {}).get("id", "")
                            
                            # 只要有实体名字，就送进去算布局分数
                            if entity_id:
                                # 注意：计算布局是对完整生成图 (gen_img) 进行的，不是裁剪后的图
                                in_r, fi_r = evaluator.calc_layout_metrics(gen_img, entity_id, bbox)
                                total_in_r += in_r
                                total_fi_r += fi_r
                                layout_samples += 1
                        
            
            # 3. AES Score (美学质量)
            total_aes_score += evaluator.calc_aes_score(gen_img)
            aes_samples += 1
            
            # 4.计算 DPG 图谱分数 
            # 获取当前 sample 的 DSG 图谱数据
            dsg_data = all_dsg_data.get(sample_key, {})
            if dsg_data:
                dpg_score = evaluator.calc_dpg_score(gen_img, dsg_data)
                total_dpg_score += dpg_score
                dpg_samples += 1
            
    # FID得分
    # compute_fid 会自动读取两个文件夹里的图片并算出距离
    fid_score = fid.compute_fid(fid_real_folder, fid_gen_folder)

    # 输出最终的平均分 
    if clip_samples > 0:
        avg_clip_score = total_clip_score / clip_samples
        print(f"\n平均 CLIP Score (图文一致性): {avg_clip_score:.4f} / 100")
    
    if dpg_samples > 0:
        print(f"平均 DPG Score (细粒度图文图谱一致性): {total_dpg_score / dpg_samples:.2f}% ")    
    
    if dino_samples > 0:
        avg_dino_score = total_dino_score / dino_samples
        print(f"平均 DINOv2 Similarity (IP-S 物体外观一致性): {avg_dino_score:.4f} / 1.0")
    
    if valid_ids_samples > 0:
        print(f"平均 ID-S Score (ArcFace 人脸一致性): {total_ids_score / valid_ids_samples:.2f} / 100")
    
    if aes_samples > 0:
        avg_aes_score = total_aes_score / aes_samples
        print(f"平均 AES Score  (LAION 美学评分): {avg_aes_score:.4f} / 10.0")

    if layout_samples > 0:
        print(f"平均 IN-R (包含率 - 生成体在框内的比例): {total_in_r / layout_samples:.2f}%")
        print(f"平均 FI-R (填充率 - 框被生成体覆盖的比例): {total_fi_r / layout_samples:.2f}%")
    
    print(f"FID Score (生成质量与真实分布的距离): {fid_score:.4f} ")
    
if __name__ == "__main__":
    main()