import torch
import torch.nn.functional as F
from torch import nn
from util import box_ops

from tqdm import tqdm

import pandas as pd
import numpy as np
import pickle


CLASSES = [
    'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
    'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
    'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]


def cls_loss(outputs, targets, criterion, cls_losses, weights=None):
    """ 

    """
    # TODO: make sure we are not backpropagating from here
    # (probably from the function that will call this(with no grad)

    outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}
    indices = criterion.matcher(outputs_without_aux, targets)

    src_idx = criterion._get_src_permutation_idx(indices)

    # order the labels by the indices
    target_classes = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)]) # BOXES
    src_logits = outputs['pred_logits'][src_idx] # (BOXES) X C 

    # order the bounding boxes by the indices
    target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0) # BOXES
    src_boxes = outputs['pred_boxes'][src_idx] # BOXES

    classes = torch.unique(target_classes)
    for cls in classes:
        idx = torch.where(target_classes == cls)[0]

        loss_ce = F.cross_entropy(src_logits[idx], target_classes[idx], reduction='sum')

        loss_bbox = F.l1_loss(src_boxes[idx], target_boxes[idx], reduction='sum')

        loss_giou = (1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes[idx]),
            box_ops.box_cxcywh_to_xyxy(target_boxes[idx])))).sum()

        losses = torch.tensor([len(idx), loss_ce, loss_bbox, loss_giou])
        cls_losses[cls] += losses

    return cls_losses
        
def create_csv(cls_losses, csv_file="cls_losses.csv"):
    df = pd.DataFrame(cls_losses.numpy())

    col_names = ["count", "loss_ce", "loss_bbox", "loss_giou", "weighted_loss"]
    mapping = {i: col_names[i] for i, _ in enumerate(col_names)}

    df = df.rename(columns=mapping)
    df["cls_name"] = CLASSES

    df = df.reindex(columns=["cls_name"] + col_names)
    df.to_csv(csv_file)
    return df


@torch.no_grad()
def cls_eval(model, criterion, data_loader, base_ds, device, output_dir):
    # evaluates the model per class
    cls_losses = torch.zeros(len(CLASSES), 4)



    model.eval()

    for i ,(samples, targets) in tqdm(enumerate(data_loader), total=len(data_loader)):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        outputs = model(samples)
        cls_loss(outputs, targets, criterion, cls_losses)


    # loss_ce, loss_bbox, loss_giou
    loss_weights = torch.tensor([1, 5, 2]).type(cls_losses.type())
    eps = torch.tensor(1e-8).type(cls_losses.type())

    weighted_loss = torch.matmul(cls_losses[:, 1: 1 + len(loss_weights)], loss_weights) / (cls_losses[:, 0] + eps)

    cls_losses = torch.cat((cls_losses, weighted_loss.unsqueeze(0).T), dim=1)

    create_csv(cls_losses)
    #np.save("cls_losses.mat", cls_losses.numpy(), pickle=True)


    print(cls_losses)
    print("finished evaluating class hardness")


    return cls_losses






