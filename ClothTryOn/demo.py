from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from PIL import Image
import torch
import numpy as np
import matplotlib.pyplot as plt

# 1. Load preprocessor (chuẩn hóa ảnh)
processor = SegformerImageProcessor.from_pretrained("./")  # thư mục chứa preprocessor_config.json

# 2. Load config + model weights
model = SegformerForSemanticSegmentation.from_pretrained(
    "./",  # thư mục chứa config.json + model.safetensors
    local_files_only=True
)
model.eval()

# Load ảnh thử nghiệm
image = Image.open("MinhDuy.jpg").convert("RGB")

# Xử lý ảnh theo preprocessor_config.json
inputs = processor(images=image, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

# logits: (batch_size, num_labels, height/4, width/4)
logits = outputs.logits  

# Resize logits về kích thước gốc của ảnh
upsampled_logits = torch.nn.functional.interpolate(
    logits,
    size=image.size[::-1],  # (height, width)
    mode="bilinear",
    align_corners=False
)

# Lấy nhãn dự đoán từng pixel
pred_seg = upsampled_logits.argmax(dim=1)[0]


# Tensor -> numpy
segmentation = pred_seg.cpu().numpy()

# Hiển thị ảnh gốc + segmentation
plt.subplot(1,2,1)
plt.imshow(image)
plt.axis("off")
plt.title("Original")

plt.subplot(1,2,2)
plt.imshow(segmentation, cmap="nipy_spectral")
plt.axis("off")
plt.title("Segmentation")

plt.show()


