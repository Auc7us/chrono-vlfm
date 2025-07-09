#!/usr/bin/env python3
import sys
import os
import torch
import math
import torch.nn.functional as F
from torch.nn import MultiheadAttention
from PIL import Image
from blip2itm_custom import Blip2ITM
from omegaconf import OmegaConf
from lavis.processors.base_processor import BaseProcessor
from lavis.common.registry import registry

def load_preprocess(preprocess_cfg):
    def _build_proc_from_cfg(cfg):
        return (
            registry.get_processor_class(cfg.name).from_config(cfg)
            if cfg is not None
            else BaseProcessor()
        )

    vis_processors = dict()
    txt_processors = dict()

    vis_proc_cfg = preprocess_cfg.get("vis_processor")
    txt_proc_cfg = preprocess_cfg.get("text_processor")

    # vis_processors["train"] = _build_proc_from_cfg(vis_proc_cfg.get("train") if vis_proc_cfg else None)
    vis_processors["eval"] = _build_proc_from_cfg(vis_proc_cfg.get("eval") if vis_proc_cfg else None)
    # txt_processors["train"] = _build_proc_from_cfg(txt_proc_cfg.get("train") if txt_proc_cfg else None)
    txt_processors["eval"] = _build_proc_from_cfg(txt_proc_cfg.get("eval") if txt_proc_cfg else None)

    return vis_processors, txt_processors

class BLIP2ITMWrapper:
    def __init__(self, name: str = "blip2_image_text_matching", model_type: str = "pretrain", device: torch.device = None):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        self.model = Blip2ITM.from_pretrained(model_type=model_type)
        self.model.eval().to(self.device)

        cfg_path = Blip2ITM.default_config_path(model_type)
        preprocess_cfg = OmegaConf.load(cfg_path).preprocess
        self.vis_processors, self.text_processors = load_preprocess(preprocess_cfg)

    def extract_feats(self, image_path: str, dummy_text: str = "room") -> torch.Tensor:
        pil_img = Image.open(image_path).convert("RGB")
        img_tensor = self.vis_processors["eval"](pil_img).unsqueeze(0).to(self.device)
        txt_input = self.text_processors["eval"](dummy_text)

        with torch.no_grad():
            out = self.model({"image": img_tensor, "text_input": txt_input}, match_head="itc")
        # returns shape (1, 32, 256) -> squeeze -> (32, 256)
        return out["image_feats"].squeeze(0) # (32, 256)

def is_image_file(fn: str) -> bool:
    return fn.lower().endswith((".jpg", ".jpeg", ".png"))

def cross_attention_fusion(
    fused_feats: torch.Tensor, new_feats:  torch.Tensor, mha: MultiheadAttention) -> torch.Tensor:
    """
    Fuse two Q-Former outputs via one cross-attention step.
    fused_feats and new_feats shape: (K=32, D=256)
    mha should be initialized with embed_dim=256, appropriate num_heads.
    """
    # add batch and seq dims: (1, 32, 256)
    Qb = fused_feats.unsqueeze(0)
    Qo = new_feats.unsqueeze(0)
    # cross-attend: queries from base, keys/values from new
    C, _ = mha(query=Qb, key=Qo, value=Qo)
    # remove batch dim: (32, 256)
    C = C.squeeze(0)
    # residual add
    return fused_feats + C

def per_query_gated_fusion(Qb: torch.Tensor, Qn: torch.Tensor) -> torch.Tensor:
    # Qb, Qn: (K, D)
    K, D = Qb.shape

    # 1) compute per-slot dot / sqrt(D)
    scores = (Qb * Qn).sum(dim=1) / math.sqrt(D)     # (K,)

    # 2) gate via sigmoid
    gates = torch.sigmoid(scores).unsqueeze(1)      # (K,1)

    # 3) novelty weight
    novelty = (1 - gates) * Qn                     # (K, D)

    # 4) residual fuse + optional layer‐norm
    fused = Qb + novelty
    return fused

def compute_cosine_similarity(feat1: torch.Tensor, feat2: torch.Tensor) -> float:
    # flatten (32,256) -> (1, 8192) -> unsqueeze (1,1,8192) and (1,8192,1)
    feat1 = F.normalize(feat1.view(1, -1), dim=-1).unsqueeze(1)  # (1, 1, 8192)
    feat2 = F.normalize(feat2.view(1, -1), dim=-1).unsqueeze(2)  # (1, 8192, 1)   
    sim = torch.bmm(feat1, feat2).mean().item()
    return sim

def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} /path/to/room_images_dir /path/to/test_images_dir")
        sys.exit(1)

    room_dir, test_dir = sys.argv[1], sys.argv[2]
    if not os.path.isdir(room_dir) or not os.path.isdir(test_dir):
        print("Both arguments must be existing directories.")
        sys.exit(1)

    itm = BLIP2ITMWrapper()
    fused_feats = None
    mha = None

    for fn in sorted(os.listdir(room_dir)):
        path = os.path.join(room_dir, fn)
        if not is_image_file(path):
            continue

        feats = itm.extract_feats(path)  # (32, 256)
        
        if fused_feats is None:
            # first view: initialize fused_feats and MHA
            fused_feats = feats.clone()
            D = fused_feats.shape[1]  # 256
            # choose num_heads dividing D
            num_heads = 8 if D % 8 == 0 else 1
            mha = MultiheadAttention(embed_dim=D, num_heads=num_heads, batch_first=True).to(itm.device)
        else:
        # cross-attention fusion options:
            # fused_feats = cross_attention_fusion(fused_feats, feats, mha)
            fused_feats = per_query_gated_fusion(fused_feats, feats)
        # other fusion options:
            # fused_feats = torch.max(fused_feats, feats)             # max-pool
            # fused_feats = (fused_feats + feats) * 0.5               # simple average
            # α = 0.2                                                 # EMA
            # fused_feats = fused_feats * (1-α) + feats * α

    if fused_feats is None:
        print("No valid images found in room directory.")
        sys.exit(1)

    for fn in sorted(os.listdir(test_dir)):
        path = os.path.join(test_dir, fn)
        if not is_image_file(path):
            continue
        test_feats = itm.extract_feats(path)  # (32, 256)
        score = compute_cosine_similarity(fused_feats, test_feats)
        print(f"{fn:50s}  cosine similarity = {score:.4f}")

if __name__ == "__main__":
    main()
