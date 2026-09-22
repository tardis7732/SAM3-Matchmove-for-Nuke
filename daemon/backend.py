"""Official SAM3 image predictor, imported only in the external CUDA process."""
from pathlib import Path
import hashlib
import numpy as np


def prepare_rgb(rgb, encoding):
    if encoding not in ('srgb', 'linear_srgb'):
        raise ValueError('Input encoding must be srgb or linear_srgb')
    rgb = np.clip(np.nan_to_num(rgb, nan=0.0, posinf=1.0, neginf=0.0), 0, 1)
    if encoding == 'linear_srgb':
        rgb = np.where(rgb <= 0.0031308, 12.92 * rgb, 1.055 * np.power(rgb, 1 / 2.4) - 0.055)
    return np.ascontiguousarray(rgb, dtype=np.float32)


def select_mask(masks, scores, index, height, width):
    if index < 0:
        raise ValueError('Object index must be nonnegative')
    masks = np.asarray(masks, dtype=bool).reshape(-1, height, width)
    scores = np.asarray(scores).reshape(-1)
    if len(scores) != len(masks) or not np.isfinite(scores).all():
        raise ValueError('Invalid SAM3 detection scores')
    areas = masks.sum(axis=(1, 2))
    order = sorted(range(len(masks)), key=lambda i: (-int(areas[i]), -float(scores[i]), i))
    if 0 <= index < len(order):
        selected = masks[order[index]]
    else:
        selected = np.zeros((height, width), dtype=bool)
    return np.ascontiguousarray(np.repeat(selected[None], 4, axis=0), dtype='<f4')


class Sam3Backend:
    def __init__(self, checkpoint):
        self.checkpoint = str(Path(checkpoint).resolve())
        if not Path(self.checkpoint).is_file():
            raise FileNotFoundError('Official sam3.pt not found: ' + self.checkpoint)
        self.processor = None
        self.image_hash = None
        self.image_state = None
        self.metrics = {}

    def predict(self, rgb, prompt, confidence, object_index):
        import torch
        from PIL import Image
        if not torch.cuda.is_available():
            raise RuntimeError('SAM3 requires the configured CUDA Python environment and NVIDIA GPU')
        if self.processor is None:
            from sam3.model_builder import build_sam3_image_model
            from sam3.model.sam3_image_processor import Sam3Processor
            model = build_sam3_image_model(checkpoint_path=self.checkpoint, load_from_HF=False, device='cuda', compile=False)
            self.processor = Sam3Processor(model, confidence_threshold=confidence)
        pixels = np.rint(rgb * 255).astype(np.uint8)
        key = (pixels.shape, hashlib.sha256(pixels.tobytes()).digest())
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            if key != self.image_hash:
                # Release the previous frame's GPU features before encoding another.
                self.image_state = None
                self.image_hash = None
                self.image_state = self.processor.set_image(Image.fromarray(pixels))
                self.image_hash = key
            self.processor.reset_all_prompts(self.image_state)
            self.processor.set_confidence_threshold(confidence)
            output = self.processor.set_text_prompt(prompt=prompt, state=self.image_state)
            masks = output['masks'].detach().cpu().numpy()
            scores = output['scores'].detach().float().cpu().numpy()
        self.metrics = {'allocated_mb': round(torch.cuda.memory_allocated() / 2**20), 'reserved_mb': round(torch.cuda.memory_reserved() / 2**20)}
        height, width = rgb.shape[:2]
        return select_mask(masks, scores, object_index, height, width), {'detections': len(scores), 'selected_index': object_index}
