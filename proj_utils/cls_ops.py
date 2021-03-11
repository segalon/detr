import torch
import torch.nn.functional as F
from torch import nn
from util import box_ops
from tqdm import tqdm



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

        losses = torch.tensor([loss_ce, loss_bbox, loss_giou])

        cls_losses[cls] += losses

    return cls_losses
        

@torch.no_grad()
def cls_eval(model, criterion, data_loader, base_ds, device, output_dir):
    # evaluates the model per class
    n_classes = 91
    cls_losses = torch.zeros(n_classes, 3)

    model.eval()

    for i ,(samples, targets) in tqdm(enumerate(data_loader), total=len(data_loader)):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        outputs = model(samples)
        loss = cls_loss(outputs, targets, criterion, cls_losses)

    np.save("cls_res.mat", cls_losses.numpy(), pickle=True)
    print("finished evaluating class hardness")

    return cls_losses






