from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
import torch
import numpy as np
from PIL import Image
from .utils import colorize_mask


_model = None
_processor = None
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




def load_model_once(model_dir: str = "model"):
    global _model, _processor
    if _model is not None and _processor is not None:
        return
    _processor = SegformerImageProcessor.from_pretrained(model_dir, local_files_only=True)
    _model = SegformerForSemanticSegmentation.from_pretrained(model_dir, local_files_only=True)
    _model.to(_device)
    _model.eval()




def _infer_mask(pil_image: Image.Image) -> np.ndarray:
    inputs = _processor(images=pil_image, return_tensors="pt")
    inputs = {k: v.to(_device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = _model(**inputs)
        logits = outputs.logits # (B, num_labels, H/4, W/4)
        upsampled = torch.nn.functional.interpolate(
        logits,
        size=pil_image.size[::-1], # (H, W)
        mode="bilinear",
        align_corners=False,
        )
        pred = upsampled.argmax(dim=1)[0].detach().cpu().numpy().astype(np.uint8)
    return pred




def predict_image(pil_image: Image.Image, overlay: bool = True, alpha: float = 0.55) -> Image.Image:
    """Return a PIL image of the segmentation result.
    If overlay=True, return original image overlaid with color mask; else return color mask only.
    """
    mask = _infer_mask(pil_image)
    color_mask = colorize_mask(mask) # RGB numpy array (H, W, 3)


    if overlay:
        base = np.array(pil_image).astype(np.float32)
        cm = color_mask.astype(np.float32)
        blended = (alpha * cm + (1 - alpha) * base).clip(0, 255).astype(np.uint8)
        return Image.fromarray(blended)
    else:
        return Image.fromarray(color_mask)