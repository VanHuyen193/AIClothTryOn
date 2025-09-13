import numpy as np


# 18-class palette for your id2label mapping (0..17)
# You can adjust colors as you like; they should be distinct.
PALETTE = [
(0, 0, 0), # 0 Background
(128, 0, 0), # 1 Hat
(255, 0, 0), # 2 Hair
(128, 64, 0), # 3 Sunglasses
(0, 128, 0), # 4 Upper-clothes
(0, 255, 0), # 5 Skirt
(0, 128, 128), # 6 Pants
(0, 255, 255), # 7 Dress
(0, 0, 128), # 8 Belt
(0, 0, 255), # 9 Left-shoe
(128, 128, 255), # 10 Right-shoe
(255, 224, 189), # 11 Face (skin tone)
(128, 0, 128), # 12 Left-leg
(255, 0, 255), # 13 Right-leg
(64, 64, 128), # 14 Left-arm
(192, 128, 128), # 15 Right-arm
(128, 128, 0), # 16 Bag
(255, 255, 0), # 17 Scarf
]




def colorize_mask(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for idx, color in enumerate(PALETTE):
        out[mask == idx] = color
    return out