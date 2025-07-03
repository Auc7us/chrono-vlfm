#!/usr/bin/env python3
import sys
import torch
import torch.nn.functional as F
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

    vis_processors["train"] = _build_proc_from_cfg(vis_proc_cfg.get("train") if vis_proc_cfg else None)
    vis_processors["eval"] = _build_proc_from_cfg(vis_proc_cfg.get("eval") if vis_proc_cfg else None)

    txt_processors["train"] = _build_proc_from_cfg(txt_proc_cfg.get("train") if txt_proc_cfg else None)
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
        return out["image_feats"]  # shape: (1, D)

def compute_cosine_similarity(feat1: torch.Tensor, feat2: torch.Tensor) -> float:
    feat1 = F.normalize(feat1.view(1, -1), dim=-1)
    feat2 = F.normalize(feat2.view(1, -1), dim=-1)
    feat1 = feat1.unsqueeze(1)  # (1, 1, 8192)
    feat2 = feat2.unsqueeze(2)  # (1, 8192, 1)    
    sim_matrix = torch.bmm(feat1, feat2)
    max_sim = sim_matrix.mean().item()

    return max_sim

def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} /path/to/image1.png /path/to/image2.png")
        sys.exit(1)

    image1_path = sys.argv[1]
    image2_path = sys.argv[2]

    itm = BLIP2ITMWrapper()
    feat1 = itm.extract_feats(image1_path)
    feat2 = itm.extract_feats(image2_path)

    sim_score = compute_cosine_similarity(feat1, feat2)
    print(f"Cosine similarity between image features: {sim_score:.4f}")

if __name__ == "__main__":
    main()
