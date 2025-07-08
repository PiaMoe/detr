# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
COCO dataset which returns image_id for evaluation.

Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
"""
from pathlib import Path

import torch
import torch.utils.data
import torchvision
from pycocotools import mask as coco_mask

import datasets.transforms as T

import os
import json
from PIL import Image
from tqdm import tqdm
from datetime import datetime



def yolo_to_coco(dataset_dir, subsets=['train', 'val', 'test'], output_dir='annotations', class_names=None):
    os.makedirs(output_dir, exist_ok=True)

    # If no class list is given, just use generic class IDs
    if class_names is None:
        class_names = []

    # Create category section
    categories = [{'id': i, 'name': name} for i, name in enumerate(class_names)]

    info = {
        "description": "",
        "url": "",
        "year": datetime.now().year,
        "contributor": "",
        "date_created": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    licenses = [{
    }]

    for subset in subsets:
        image_id = 0
        annotation_id = 0
        coco_output = {
            'info': info,
            'licenses': licenses,
            'images': [],
            'annotations': [],
            'categories': categories
        }

        label_dir = os.path.join(dataset_dir, subset, 'labels')
        image_dir = os.path.join(dataset_dir, subset, 'images')
        label_files = [f for f in os.listdir(label_dir) if f.endswith('.txt')]

        for label_file in tqdm(label_files, desc=f"Processing {subset}"):
            image_file = os.path.splitext(label_file)[0] + '.jpg'
            image_path = os.path.join(image_dir, image_file)
            if not os.path.exists(image_path):
                image_file = os.path.splitext(label_file)[0] + '.png'
                image_path = os.path.join(image_dir, image_file)
                if not os.path.exists(image_path):
                    continue  # Skip if no corresponding image

            # Get image size
            with Image.open(image_path) as img:
                width, height = img.size

            # Add image entry
            coco_output['images'].append({
                'id': image_id,
                'file_name': image_file,
                'width': width,
                'height': height
            })

            # Read annotations
            with open(os.path.join(label_dir, label_file), 'r') as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    class_id, x_center, y_center, w, h = map(float, parts)
                    class_id = int(class_id)
                    x_center *= width
                    y_center *= height
                    w *= width
                    h *= height
                    x_min = x_center - w / 2
                    y_min = y_center - h / 2

                    coco_output['annotations'].append({
                        'id': annotation_id,
                        'image_id': image_id,
                        'category_id': class_id,
                        'bbox': [x_min, y_min, w, h],
                        'area': w * h,
                        'iscrowd': 0
                    })
                    annotation_id += 1

            image_id += 1

        # Save JSON
        with open(os.path.join(output_dir, f'{subset}.json'), 'w') as f:
            json.dump(coco_output, f, indent=2)



class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_file, transforms, return_masks):
        super(CocoDetection, self).__init__(img_folder, ann_file)
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks)

    def __getitem__(self, idx):
        img, target = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        img, target = self.prepare(img, target)
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        return img, target


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        return image, target


def make_coco_transforms(image_set):

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.RandomResize(scales, max_size=1333),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales, max_size=1333),
                ])
            ),
            normalize,
        ])

    if image_set == 'val':
        return T.Compose([
            T.RandomResize([800], max_size=1333),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def build(image_set, args):
    root = Path(args.data_path)
    assert root.exists(), f'provided data path {root} does not exist'
    mode = 'instances'
    PATHS = {
        "train": (root / "train/images", root / "train" / f'train.json'),
        "val": (root / "val/images", root / "val" / f'val.json'),
    }

    img_folder, ann_file = PATHS[image_set]
    dataset = CocoDetection(img_folder, ann_file, transforms=make_coco_transforms(image_set), return_masks=args.masks)
    return dataset



if __name__ == "__main__":

    base_dir = '../../../data/BOArDING_Dataset/BOArDING_Det'

    # Example usage
    yolo_to_coco(
        dataset_dir=base_dir,
        subsets=['train', 'val', 'test'],
        output_dir=base_dir + '/annotations',
        class_names=['boat']
    )
