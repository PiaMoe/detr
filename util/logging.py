import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as T
import torch


def visualize_prediction(samples, outputs):
    # visualize first 3 images in batch
    images = []
    num_images = min(3, samples.tensors.shape[0])
    for idx in range(num_images):
        image_tensor = samples.tensors[idx].cpu()

        # undo normalization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image_tensor = image_tensor * std + mean  # Undo normalization
        image_tensor = torch.clamp(image_tensor, 0, 1)

        image_pil = T.ToPILImage()(image_tensor)

        prob = outputs["pred_logits"].softmax(-1)[idx]
        boxes = outputs["pred_boxes"][idx]
        score_threshold = 0.7
        keep = prob.max(-1).values > score_threshold

        probs = prob[keep]
        boxes = boxes[keep]

        draw = ImageDraw.Draw(image_pil)
        font = ImageFont.load_default()
        w, h = image_pil.size

        for p, box in zip(probs, boxes):
            cls = p.argmax().item()
            score = p.max().item()
            label = f"class {cls}: {score:.2f}"

            cx, cy, bw, bh = box
            x0 = (cx - bw / 2.0) * w
            y0 = (cy - bh / 2.0) * h
            x1 = (cx + bw / 2.0) * w
            y1 = (cy + bh / 2.0) * h
            x0, x1 = sorted([x0.item(), x1.item()])
            y0, y1 = sorted([y0.item(), y1.item()])
            draw.rectangle([x0, y0, x1, y1], outline="red", width=2)
            draw.text((x0 + 4, y0 + 4), label, fill="white", font=font)

        images.append(image_pil)

    return images
