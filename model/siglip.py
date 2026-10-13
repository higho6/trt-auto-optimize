from PIL import Image
import requests
from transformers import AutoImageProcessor, SiglipVisionModel
import torch

model = SiglipVisionModel.from_pretrained("/workspace/ckpt/siglip-base-patch16-224").eval()
processor = AutoImageProcessor.from_pretrained("/workspace/ckpt/siglip-base-patch16-224")

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
with requests.get(url, stream=True, timeout=30) as response:
    response.raise_for_status()
    with Image.open(response.raw) as source_image:
        image = source_image.convert("RGB")

inputs = processor(images=image, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

print("last_hidden_state:", outputs.last_hidden_state.shape)
print("pooler_output:", outputs.pooler_output.shape)