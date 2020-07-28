import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def focal_loss(logits, targets, alpha, gamma, normalizer):
    """Compute the focal loss between `logits` and the golden `target` values.

    Focal loss = -(1-pt)^gamma * log(pt)
    where pt is the probability of being classified to the true class.

    Args:
        logits: A float32 tensor of size [batch, height_in, width_in, num_predictions].

        targets: A float32 tensor of size [batch, height_in, width_in, num_predictions].

        alpha: A float32 scalar multiplying alpha to the loss from positive examples
            and (1-alpha) to the loss from negative examples.

        gamma: A float32 scalar modulating loss from hard and easy examples.

         normalizer: A float32 scalar normalizes the total loss from all examples.

    Returns:
        loss: A float32 scalar representing normalized total loss.
    """

    positive_label_mask = targets == 1.0
    cross_entropy = F.binary_cross_entropy_with_logits(logits, targets.to(logits.dtype), reduction='none')
    # Below are comments/derivations for computing modulator.
    # For brevity, let x = logits,  z = targets, r = gamma, and p_t = sigmod(x)
    # for positive samples and 1 - sigmoid(x) for negative examples.
    #
    # The modulator, defined as (1 - P_t)^r, is a critical part in focal loss
    # computation. For r > 0, it puts more weights on hard examples, and less
    # weights on easier ones. However if it is directly computed as (1 - P_t)^r,
    # its back-propagation is not stable when r < 1. The implementation here
    # resolves the issue.
    #
    # For positive samples (labels being 1),
    #    (1 - p_t)^r
    #  = (1 - sigmoid(x))^r
    #  = (1 - (1 / (1 + exp(-x))))^r
    #  = (exp(-x) / (1 + exp(-x)))^r
    #  = exp(log((exp(-x) / (1 + exp(-x)))^r))
    #  = exp(r * log(exp(-x)) - r * log(1 + exp(-x)))
    #  = exp(- r * x - r * log(1 + exp(-x)))
    #
    # For negative samples (labels being 0),
    #    (1 - p_t)^r
    #  = (sigmoid(x))^r
    #  = (1 / (1 + exp(-x)))^r
    #  = exp(log((1 / (1 + exp(-x)))^r))
    #  = exp(-r * log(1 + exp(-x)))
    #
    # Therefore one unified form for positive (z = 1) and negative (z = 0)
    # samples is:
    #      (1 - p_t)^r = exp(-r * z * x - r * log(1 + exp(-x))).
    neg_logits = -1.0 * logits
    modulator = torch.exp(gamma * targets * neg_logits - gamma * torch.log1p(torch.exp(neg_logits)))
    loss = modulator * cross_entropy
    weighted_loss = torch.where(positive_label_mask, alpha * loss, (1.0 - alpha) * loss)
    weighted_loss /= normalizer
    return weighted_loss


def bbox_iou(box1, box2, x1y1x2y2=False, mask= None, ltype = "GIOU"):
    # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
    GIoU = False
    CIoU = False
    DIoU = False
    if ltype == "GIOU":
        GIoU = True
    elif ltype == "CIOU":
        CIoU = True
    elif ltype == "DIOU":
        DIoU = True
    else:
        assert(0)
    
    box2 = box2.t()
    if mask != None:
        box1 *= mask
    box1 = box1.t().float()
    # Get the coordinates of bounding boxes
    if x1y1x2y2:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[1], box1[0], box1[3], box1[2]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[1], box2[0], box2[3], box2[2]
    else:  # transform from xywh to xyxy ty, tx, th, tw = rel_codes
        b1_x1, b1_x2 = box1[1] - torch.exp(box1[3]) / 2, box1[1] + torch.exp(box1[3]) / 2
        b1_y1, b1_y2 = box1[0] - torch.exp(box1[2]) / 2, box1[0] + torch.exp(box1[2]) / 2
        b2_x1, b2_x2 = box2[1] - torch.exp(box2[3]) / 2, box2[1] + torch.exp(box2[3]) / 2
        b2_y1, b2_y2 = box2[0] - torch.exp(box2[2]) / 2, box2[0] + torch.exp(box2[2]) / 2

    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1
    #w1 = w1.abs()#torch.clamp(w1,min = -0.1,max = 0.1)
    #h1 = h1.abs()#torch.clamp(h1,min = -0.1,max = 0.1)
    #w2 = w2.abs()#torch.clamp(w2,min = -0.1,max = 0.1)
    #h2 = h2.abs()#torch.clamp(h2,min = -0.1,max = 0.1)
    area1 = w1*h1
    area2 = w2*h2
    union = area1+area2+1e-16 - inter

    iou = inter / union  # iou
    if GIoU or DIoU or CIoU:
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex (smallest enclosing box) width
        
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        
        if GIoU:  # Generalized IoU https://arxiv.org/pdf/1902.09630.pdf
            c_area = cw * ch + 1e-16  # convex area
            return iou - (c_area - union) / c_area  # GIoU
        if DIoU or CIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            # convex diagonal squared
            c2 = cw ** 2 + ch ** 2 + 1e-16
            # centerpoint distance squared
            rho2 = ((b2_x1 + b2_x2) - (b1_x1 + b1_x2)) ** 2 / 4 + ((b2_y1 + b2_y2) - (b1_y1 + b1_y2)) ** 2 / 4
            if DIoU:
                return iou - rho2 / c2  # DIoU
            elif CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    alpha = v / (1 - iou + v+1e-16)
                return iou - (rho2 / c2 + v * alpha)  # CIoU
    #if mask != None:
     #   iou*= mask
    return iou


def giou_loss(box_outputs,box_targets_at_level,num_positives,loss_type):
    #nb = box_outputs.shape[0]
    #pxy = box_outputs[:, :2].sigmoid() * 2. - 0.5
    #pwh = (box_outputs[:, 2:4].sigmoid() * 2) ** 2 
    #pbox = torch.cat((pxy, pwh), 1).to('cuda')  # predicted box
    box_targets = box_targets_at_level.view([-1,4])
    mask = box_targets != 0.0
    giou = bbox_iou(box_outputs.reshape([-1,4]), box_targets, x1y1x2y2=False,mask = mask,ltype= loss_type)
    
    return (1.0 - giou).sum()/(num_positives*8.0)#/num_positives#.mean()


def huber_loss(input, target, delta=1., weights=None, size_average=True):
    """
    """
    err = input - target
    abs_err = err.abs()
    quadratic = torch.clamp(abs_err, max=delta)
    linear = abs_err - quadratic
    loss = 0.5 * quadratic.pow(2) + delta * linear
    if weights is not None:
        loss *= weights
    return loss.mean() if size_average else loss.sum()


def smooth_l1_loss(input, target, beta=1. / 9, weights=None, size_average=True):
    """
    very similar to the smooth_l1_loss from pytorch, but with the extra beta parameter
    """
    if beta < 1e-5:
        # if beta == 0, then torch.where will result in nan gradients when
        # the chain rule is applied due to pytorch implementation details
        # (the False branch "0.5 * n ** 2 / 0" has an incoming gradient of
        # zeros, rather than "no gradient"). To avoid this issue, we define
        # small values of beta to be exactly l1 loss.
        loss = torch.abs(input - target)
    else:
        err = torch.abs(input - target)
        loss = torch.where(err < beta, 0.5 * err.pow(2) / beta, err - 0.5 * beta)
    if weights is not None:
        loss *= weights
    return loss.mean() if size_average else loss.sum()


def _classification_loss(cls_outputs, cls_targets, num_positives, alpha=0.25, gamma=2.0):
    """Computes classification loss."""
    normalizer = num_positives
    classification_loss = focal_loss(cls_outputs, cls_targets, alpha, gamma, normalizer)
    return classification_loss


def _box_loss(box_outputs, box_targets, num_positives, delta=0.1):
    """Computes box regression loss."""
    # delta is typically around the mean value of regression target.
    # for instances, the regression targets of 512x512 input with 6 anchors on
    # P3-P7 pyramid is about [0.1, 0.1, 0.2, 0.2].
    normalizer = num_positives * 4.0
    mask = box_targets != 0.0
    box_loss = huber_loss(box_targets, box_outputs, weights=mask, delta=delta, size_average=False)
    box_loss /= normalizer
    return box_loss


class DetectionLoss(nn.Module):
    def __init__(self, config):
        super(DetectionLoss, self).__init__()
        self.config = config
        self.num_classes = config.num_classes
        self.alpha = config.alpha
        self.gamma = config.gamma
        self.delta = config.delta
        self.box_loss_weight = config.box_loss_weight

    def forward(self, cls_outputs, box_outputs, cls_targets, box_targets, num_positives):
        """Computes total detection loss.
        Computes total detection loss including box and class loss from all levels.
        Args:
            cls_outputs: an OrderDict with keys representing levels and values
                representing logits in [batch_size, height, width, num_anchors].

            box_outputs: an OrderDict with keys representing levels and values
                representing box regression targets in [batch_size, height, width, num_anchors * 4].

            cls_targets: groundtruth class targets.

            box_targets: groundtrusth box targets.

            num_positives: num positive grountruth anchors

        Returns:
            total_loss: an integer tensor representing total loss reducing from class and box losses from all levels.

            cls_loss: an integer tensor representing total class loss.

            box_loss: an integer tensor representing total box regression loss.
        """
        if isinstance(num_positives, list):
            # if num_positives is a list, all targets assumed to be batch size lists of tensors (or level->tensors)
            stack_targets = True
            num_positives = torch.stack(num_positives)
        else:
            # targets are already tensors
            stack_targets = False

        # Sum all positives in a batch for normalization and avoid zero
        # num_positives_sum, which would lead to inf loss during training
        num_positives_sum = num_positives.sum() + 1.0
        levels = len(cls_outputs)
        
                
        
        cls_losses = []
        box_losses = []
        new_box_losses = []
        for l in range(levels):
            if stack_targets:
                cls_targets_at_level = torch.stack([b[l] for b in cls_targets])
                box_targets_at_level = torch.stack([b[l] for b in box_targets])
            else:
                cls_targets_at_level = cls_targets[l]
                box_targets_at_level = box_targets[l]

            # Onehot encoding for classification labels.
            # NOTE: PyTorch one-hot does not handle -ve entries (no hot) like Tensorflow, so mask them out
            cls_targets_non_neg = cls_targets_at_level >= 0
            cls_targets_at_level_oh = F.one_hot(cls_targets_at_level * cls_targets_non_neg, self.num_classes)
            cls_targets_at_level_oh = torch.where(
               cls_targets_non_neg.unsqueeze(-1), cls_targets_at_level_oh, torch.zeros_like(cls_targets_at_level_oh))

            bs, height, width, _, _ = cls_targets_at_level_oh.shape
            cls_targets_at_level_oh = cls_targets_at_level_oh.view(bs, height, width, -1)
            cls_loss = _classification_loss(
                cls_outputs[l].permute(0, 2, 3, 1),
                cls_targets_at_level_oh,
                num_positives_sum,
                alpha=self.alpha, gamma=self.gamma)
            cls_loss = cls_loss.view(bs, height, width, -1, self.num_classes)
            cls_loss *= (cls_targets_at_level != -2).unsqueeze(-1).float()
            cls_losses.append(cls_loss.sum())
            
            if self.config.loss_type == "HUBER":
                box_losses.append(_box_loss(
                    box_outputs[l].permute(0, 2, 3, 1),# batchsize,w,h,number of boxes
                    box_targets_at_level,
                    num_positives_sum,
                    delta=self.delta))
            else:
                box_losses.append(giou_loss(
                    box_outputs[l].permute(0, 2, 3, 1),# batchsize,w,h,number of boxes
                    box_targets_at_level,
                    num_positives_sum,
                    loss_type = self.config.loss_type
                    ))

        # Sum per level losses to total loss.
        cls_loss = torch.sum(torch.stack(cls_losses, dim=-1), dim=-1)
        box_loss = torch.sum(torch.stack(box_losses, dim=-1), dim=-1)
        total_loss = cls_loss + self.box_loss_weight * box_loss
        return total_loss, cls_loss, box_loss

