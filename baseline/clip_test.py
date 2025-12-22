import json
import os
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from tqdm import tqdm

CLIP_MODEL_PATH = "model/clip"  # 本地 CLIP 模型路径
# PRE_path = ["data/data_okvqa", "data/data_1017", "data/local_data"]
PRE_path = ["data/data_okvqa"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# 2. 加载 CLIP
# =========================
model = CLIPModel.from_pretrained(CLIP_MODEL_PATH).to(DEVICE)
processor = CLIPProcessor.from_pretrained(CLIP_MODEL_PATH)
model.eval()

def compute_image_similarity(img1_path, img2_path):
    try:
        img1 = Image.open(img1_path).convert("RGB")
        img2 = Image.open(img2_path).convert("RGB")
    except Exception as e:
        print(f"[WARN] Image load failed: {e}")
        return None

    inputs = processor(images=[img1, img2], return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        feats = model.get_image_features(**inputs)
        feats = F.normalize(feats, dim=-1)
        similarity = torch.matmul(feats[0], feats[1]).item()

    return similarity
# =========================
# 1. 路径配置
# =========================
def test():
    for p_path in PRE_path:
        INPUT_JSON = f"{p_path}/input.json"
        OUTPUT_JSON = f"{p_path}/input_with_similarity.json"

        OKVQA_IMAGE_DIR = f"{p_path}/image"
        SCREENSHOT_DIR = f"{p_path}/screenshot"

        # =========================
        # 4. 读取 JSON
        # =========================
        with open(INPUT_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert isinstance(data, list), "JSON must be a list of samples"

        # =========================
        # 5. 主处理循环
        # =========================
        new_data = []

        for sample in tqdm(data):
            image_filename = f"{sample['image_id']}.jpg"  # 保留 .jpg
            image_stem, _ = os.path.splitext(image_filename)

            img1_path = os.path.join(OKVQA_IMAGE_DIR, image_filename)
            img2_path = os.path.join(SCREENSHOT_DIR, f"{image_stem}-search.png")

            if not (os.path.exists(img1_path) and os.path.exists(img2_path)):
                print(f"[WARN] Missing image for image_id={image_filename}")
                similarity = None
            else:
                similarity = compute_image_similarity(img1_path, img2_path)

            # 保持原字段不变，仅新增 similarity
            new_sample = dict(sample)
            new_sample["similarity"] = similarity

            new_data.append(new_sample)

        # =========================
        # 6. 保存新 JSON
        # =========================
        with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
            json.dump(new_data, f, ensure_ascii=False, indent=2)

        print(f"Saved to {OUTPUT_JSON}")

test()