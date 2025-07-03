import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from torch.cuda.amp import autocast
from huggingface_hub import snapshot_download
from PIL import Image

# 1) Download & locate the model cache (just to verify)
model_id = "Salesforce/blip2-flan-t5-xl"
path = snapshot_download(model_id)
print("Model files at:", path)

# 2) Load processor + model (FP16, auto device map)
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = Blip2Processor.from_pretrained(model_id)
model = Blip2ForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto",      # will spill to CPU if GPU memory is tight
)
model.eval()

my_prompt = (
    "Is there a dog infront? Reply using yes or no"
)

# 3) Inference
raw_image = Image.open("/home/chrono-user/mountdir/images/t1.png").convert("RGB")
inputs = processor(images=raw_image,
                   text=my_prompt,
                   return_tensors="pt").to(device)

with autocast():
    outputs = model.generate(**inputs, no_repeat_ngram_size=2)

caption = processor.batch_decode(outputs, skip_special_tokens=True)[0]
print("Caption:", caption)