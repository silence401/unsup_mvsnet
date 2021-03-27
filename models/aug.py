import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
def random_image_mask(img, filter_size):
    '''

    :param img: [B x 3 x H x W]
    :param crop_size:
    :return:
    '''
    fh, fw = filter_size
    _, _, h, w = img.size()

    if fh == h and fw == w:
        return img, None

    x = np.random.randint(0, w - fw)
    y = np.random.randint(0, h - fh)
    filter_mask = torch.ones_like(img)   # B x 3 x H x W
    filter_mask[:, :, y:y+fh, x:x+fw] = 0.0    # B x 3 x H x W
    img = img * filter_mask    # B x 3 x H x W
    return img, filter_mask


# def aug_loss(depth_est, depth_gt, mask):
#     print("mask", mask)
#     print("depth_est", depth_est)
#     print("depth_gt", depth_gt)
#     mask = mask > 0.5
#     return F.smooth_l1_loss(depth_est[mask], depth_gt[mask], reduction='mean')
class AugLoss(nn.Module):
    def __init__(self, **kwargs):
        super(AugLoss, self).__init__()
    
    def forward(self, depth_est, depth_gt, mask):
        mask = mask > 0.5
        #print("mask", mask)
        return F.smooth_l1_loss(depth_est[mask], depth_gt[mask], reduction='mean')