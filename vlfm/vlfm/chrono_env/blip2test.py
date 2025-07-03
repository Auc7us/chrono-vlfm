import torch
import torch.nn.functional as F
from PIL import Image
from transformers import Blip2Processor, Blip2ForImageTextRetrieval

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load processor and model
processor = Blip2Processor.from_pretrained("Salesforce/blip2-itm-vit-g")
model = Blip2ForImageTextRetrieval.from_pretrained("Salesforce/blip2-itm-vit-g").to(device).eval()

def extract_and_score_hf(image_path: str, prompt: str):
    # Preprocess image and text
    raw_image = Image.open(image_path).convert("RGB")
    inputs = processor(images=raw_image, text=prompt, return_tensors="pt").to(device)

    # Forward pass to get cosine similarity
    with torch.no_grad():
        outputs = model(**inputs, return_dict=True)
        image_embed = outputs.image_embeds.mean(dim=1)  # [1, D]
        text_embed  = outputs.text_embeds.mean(dim=1)   # [1, D]

        # Cosine similarity
        score = F.cosine_similarity(image_embed, text_embed).item()


        # Manually extract Q-Former embeddings
        # 1. Vision encoding
        vision_outputs = model.vision_model(pixel_values=inputs["pixel_values"])
        image_embeds = vision_outputs.last_hidden_state

        # 2. Q-Former input prep
        query_tokens = model.query_tokens.expand(image_embeds.shape[0], -1, -1)
        qformer_outputs = model.qformer(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=torch.ones(image_embeds.shape[:-1], dtype=torch.long).to(device),
            return_dict=True,
        )
        qformer_embeds = qformer_outputs.last_hidden_state

    return qformer_embeds, score


# Example
q_feats, cosine_score = extract_and_score_hf(
    "/home/chrono-user/mountdir/images/t1.png",
    "Seems like there is a tv on the dresser"
)
print("Q-Former embeddings:", q_feats.shape)   # e.g. (1, 32, 1024)
print(f"Cosine (ITC) score: {cosine_score:.4f}")
