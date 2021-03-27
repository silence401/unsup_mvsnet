import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from .module import *
#test#
#import sys
#sys.path.append('/home/silence401/unsup_cascade-stereo/CasMVSNet/')
from torchvision.transforms import Resize
from config import args
#待删
import cv2
import matplotlib.pyplot as plt


class UnSupLoss(nn.Module):
    def __init__(self, args, net=None, **kwargs):
        super(UnSupLoss, self).__init__()
        self.satw = SaTWNet(base_channels=8, num_stage=3)
        self.ssim = SSIM()
        self.stage_infos = {
            "stage1":{
                "scale": 4.0,
            },
            "stage2": {
                "scale": 2.0,
            },
            "stage3": {
                "scale": 1.0,
            }
        }
        self.depth_loss_weight = kwargs.get("dlossw", None)
        
    def forward(self, imgs, cams, outputs):
        index = 0
        #imgs [{stage1, stage2, stage3},{}]
        #depth_loss_weight = args.get("dlossw", None)
        #depth_loss_weight = None
        #print("============start losss==============")
        # print(imgs[0].shape)
        # print(type(imgs))
        # print(imgs[0]['stage1'].shape)
        # B, num_views, _, h, w  = imgs.shape 
        ref_idx = 0
        loss_photo = 0
        loss_ssim = 0
        loss_smooth = 0
        _, num_views, _, _, _ = cams['stage1'][:].shape
        #print(cams['stage1'].shape)
        weights = outputs['weights']
        #print(imgs.shape)
        ref_img = imgs[:, ref_idx]


        for view in range(1, num_views):   
            #print("===========")
            src_img = imgs[:, view]
            for (stage_inputs, stage_key) in [(outputs[k], k) for k in outputs.keys() if "stage" in k]:
                stage_idx = int(stage_key.replace("stage", "")) - 1
                
                ref_cam = cams[stage_key][:, ref_idx]
                src_cam = cams[stage_key][:, view]
                
                
                ref_img_stage = F.interpolate(ref_img, scale_factor=1/self.stage_infos[stage_key]['scale'], mode='nearest')
                src_img_stage = F.interpolate(src_img, scale_factor=1/self.stage_infos[stage_key]['scale'], mode='nearest')
              
                #ref_img_stage = ref_img_stage.permute(0, 3, 1, 2)
                #src_img_stage = src_img_stage.permute(0, 3, 1, 2)
                
                
#                 print("FFFFFFFFFFFFFFFFFFFFFFFFFf")
#                 print("ref_img.shape:", ref_img_stage.shape)
                
                
                
                depth = stage_inputs['depth']
                #depth = depth_gt[stage_key]
                #print('depth.shape:', depth.shape)
                #depth2 = torch.zeros_like(depth).to(depth.device)
                #depth2 = torch.randn(depth.shape).to(depth.device)
                
                #depth_gd = depthgd[stage_key][0]
                #plt.imsave('depth'+stage_key+'.png', depth.cpu().detach().numpy().squeeze(0))
                #plt.imsave('depth.png', depth.cpu().detach().numpy().squeeze(0))
                
                
               
                #print("depth_test", depth_test[stage_key].shape)
                weights_stage = F.interpolate(weights, scale_factor=1/self.stage_infos[stage_key]['scale'], mode='nearest')
                weight = weights_stage[:, view-1].unsqueeze(1)
                #print("weigth_stage", weight.shape)
                warped_src_img, mask = inverse_warping(src_img_stage, ref_cam, src_cam, depth)
                
                #loss_mask = compute_mask_loss(mask)
                
               # print("warped_src_img.shape:", warped_src_img.shape)
                #print(F.smooth_l1_loss(warped_src_img*mask, depth_test[s))
                
                #print("warped_src_img.shape", warped_src_img.shape)
                #print("mask.shape", mask.shape)
                #print(ref_img.shape)
#                 if stage_idx == 2:
#                     plt.imsave('ref_img'+stage_key+'.png', ref_img.cpu().detach().numpy().squeeze(0).transpose(1, 2, 0))
#                     plt.imsave('depth'+stage_key+'.png', depth.cpu().detach().numpy().squeeze(0))
#                     plt.imsave('mask'+stage_key+'.png', mask.cpu().detach().numpy().squeeze(0).squeeze(0))
#                     plt.imsave('warped_src'+stage_key+'.png', warped_src_img.cpu().detach().numpy().squeeze(0).transpose(1, 2, 0))
                #cv2.imwrite('src_img'+stage_key+'_{:d}.png'.format(index), src_img.cpu().detach().numpy().squeeze(0).transpose(1, 2, 0)*255)
                #cv2.imwrite('mask'+stage_key+'_{:d}.png'.format(1), mask.cpu().detach().numpy()[0].transpose(1, 2, 0)*255)
                #cv2.imwrite('mask'+stage_key+'_{:d}.png'.format(2), mask.cpu().detach().numpy()[1].transpose(1, 2, 0)*255)
                #print("save completed")
                if self.depth_loss_weight is not None:
                    #print(self.depth_loss_weight[stage_idx])
                    loss_photo += self.depth_loss_weight[stage_idx]*compute_photo_loss(weight, warped_src_img, ref_img_stage, mask,simple=False)
                    loss_ssim += self.depth_loss_weight[stage_idx]*torch.mean(self.ssim(ref_img_stage, warped_src_img, mask))
                else:
                    loss_photo += 1.0*compute_photo_loss(weight, warped_src_img, ref_img_stage, mask,simple=False)
                    loss_ssim += 1.0*torch.mean(self.ssim(ref_img_stage, warped_src_img, mask))


                if view == 1 and stage_idx == 2:
                    loss_smooth += depth_smoothness(depth.unsqueeze(dim=1), ref_img_stage, args.smooth_lambda)
#                 print("============stage_key{}======".format(stage_key))
#                 print(loss_photo)
#                 print(loss_ssim)
#                 print(loss_smooth)
                    
                

#         print("loss_photo:", 50*loss_photo)
#         print("loss_ssim:",  1*loss_ssim)
#         print("loss_smmoth:", 1*loss_smooth)
#         print("loss_mask:", 6*loss_mask)
        print("5*loss_photo + 0.3*loss_ssim + 0.018*loss_smooth:", 50*loss_photo + 1*loss_ssim + 1*loss_smooth)
        #return 50*loss_photo + 6*loss_ssim + 0.18*loss_smooth, loss_photo, loss_ssim, loss_smooth
        return 80*loss_photo + 2*loss_ssim + 0.1*loss_smooth, loss_photo, loss_ssim, loss_smooth

#change the [{}, {}] to {}
# def update_dict(cams, idx):
#     ref_cam = {"stage1": cams["stage1"][:, 0]}
#     ref_cam.update({"stage2": cams["stage2"][:, 0]})
#     ref_cam.update({"stage3": cams["stage3"][:, 0]})
    
#     src_cam = {"stage1": cams["stage1"][:, idx]}
#     src_cam.update({"stage2": cams["stage2"][:, idx]})
#     src_cam.update({"stage3": cams["stage3"][:, idx]})

#     return ref_cam, src_cam
    
def update_dict(origin_dict, idx):
    new_dict = {"stage1": origin_dict["stage1"][:, idx]}
    new_dict.update({"stage2": origin_dict["stage2"][:, idx]})
    new_dict.update({"stage3": origin_dict["stage3"][:, idx]})
    return new_dict

def update_depth_dict(origin_dict):
    new_dict = {"stage1": origin_dict["stage1"]['depth']}
    new_dict.update({"stage2": origin_dict["stage2"]['depth']})
    new_dict.update({"stage3": origin_dict["stage3"]['depth']})
    return new_dict


class SSIM(nn.Module):
    """Layer to compute the SSIM loss between a pair of images
    """
    def __init__(self):
        super(SSIM, self).__init__()
        self.mu_x_pool   = nn.AvgPool2d(3, 1)
        self.mu_y_pool   = nn.AvgPool2d(3, 1)
        self.sig_x_pool  = nn.AvgPool2d(3, 1)
        self.sig_y_pool  = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)
        self.mask_pool = nn.AvgPool2d(3, 1)
        # self.refl = nn.ReflectionPad2d(1)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y, mask):
       # print("X", x.shape)
       # print("Y", y.shape)
        # print('mask: {}'.format(mask.shape))
        # print('x: {}'.format(x.shape))
        # print('y: {}'.format(y.shape))
        #x = x.permute(0, 3, 1, 2)  # [B, H, W, C] --> [B, C, H, W]
        #y = y.permute(0, 3, 1, 2)
       # mask = mask.permute(0, 3, 1, 2)

        # x = self.refl(x)
        # y = self.refl(y)
        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)
        sigma_x  = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y  = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y
        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)
        SSIM_mask = self.mask_pool(mask)
        output = SSIM_mask * torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)
        #return output.permute(0, 2, 3, 1)  # [B, C, H, W] --> [B, H, W, C]
        #print("ssim output", output.shape)
        return output
    
def gradient_x(img):
    #print("img:", img.shape)
    return img[:, :, :-1, :] - img[:, :, 1:, :]

def gradient_y(img):
    return img[:, :, :, :-1] - img[:, :, :, 1:]

def gradient(pred):
    #print("pred:", pred.shape)
    D_dy = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    D_dx = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    return D_dx, D_dy

def depth_smoothness(depth, img, lambda_wt=1):
    """Computes image-aware depth smoothness loss."""
    #print("depth_startshape:", depth.shape)
    depth_dx = gradient_x(depth)
    depth_dy = gradient_y(depth)
    image_dx = gradient_x(img)
    image_dy = gradient_y(img)
    #plt.imsave("image_dx.png", image_dx.cpu().detach().squeeze(0).permute(1, 2, 0).numpy())
    #plt.imsave("image_dy.png", image_dy.cpu().detach().squeeze(0).permute(1, 2, 0).numpy())
    weights_x = torch.exp(-(lambda_wt * torch.mean(torch.abs(image_dx), 1, keepdim=True)))
    weights_y = torch.exp(-(lambda_wt * torch.mean(torch.abs(image_dy), 1, keepdim=True)))
   # print("depth_dx:", depth_dx.shape)
   # print("weight_dx:", weights_x.shape)
    smoothness_x = depth_dx * weights_x
    smoothness_y = depth_dy * weights_y
    return torch.mean(torch.abs(smoothness_x)) + torch.mean(torch.abs(smoothness_y))


def compute_photo_loss(weight, warped, ref, mask, simple=True):
    #print("weights:",weights.shape)
   # print("warped:", warped.shape)
    if simple:
        return F.smooth_l1_loss(warped*mask, ref*mask, reduction='mean')
    else:
        
        alpha = 0.5
        ref_dx, ref_dy = gradient(ref * mask)
        #print("ref_dx.shape:", ref_dx.shape)
        #plt.imsave('ref_dx.png', ref_dx.cpu().detach().squeeze(0).permute(1, 2, 0).numpy())
        #plt.imsave('ref_dy.png', ref_dy.cpu().detach().squeeze(0).permute(1, 2, 0).numpy())
        warped_dx, warped_dy = gradient(warped * mask)
        #plt.imsave('warped_dx.png', warped_dx.cpu().detach().squeeze(0).permute(1, 2, 0).numpy())
       # plt.imsave('warped_dy.png', warped_dy.cpu().detach().squeeze(0).permute(1, 2, 0).numpy())
        photo_loss = F.smooth_l1_loss(weight*warped*mask, weight*ref*mask, reduction='mean')
        #TODO weight can be add in grad_loss 
        grad_loss = F.smooth_l1_loss(warped_dx, ref_dx, reduction='mean') + \
                    F.smooth_l1_loss(warped_dy, ref_dy, reduction='mean')
        return (1 - alpha) * photo_loss + alpha * grad_loss

def compute_mask_loss(mask):
    #print("mask.shape:", mask.shape)
    b, _, h, w = mask.shape
    sum_ = torch.ones((b,1,h,w)).to(mask.device)
#     mask_ = mask.reshape((b,1,-1))
#     mask_ = torch.sum(mask_, dim = 2)
#     sum_ = torch.tensor(h*w).to(mask.device)
#     sum_ = sum_.repeat(b,1,1)
# #     print("sum:", sum_.shape)
# #     print(sum_.float() - mask_.float())
#     return torch.mean((sum_.float() - mask_.float())/h/w)
    return F.smooth_l1_loss(sum_, mask, reduction='mean')

    

if __name__ == '__main__':

    from datasets.dtu_yao import MVSDataset
    view_num = 3
    depth_num = 192
    depth_interval = 1.06

    datapath = "/home/silence401/下载/dataset/mvsnet/dtu_training"
    listfile = "./lists/dtu/train.txt"

    train_dataset = MVSDataset(datapath, listfile, "train", nviews=view_num, ndepths=depth_num, interval_scale=depth_interval)
    item  = train_dataset[100]
    print(item.keys())

    ref_img = torch.tensor(item['imgs'][0]).unsqueeze(0)
    ref_img_gradx = gradient_x(ref_img)
    ref_img_gradx = ref_img_gradx.squeeze(0)
    ref_img_gradx = ref_img_gradx.permute(1, 2, 0)
    
    from matplotlib import pyplot as plt
    plt.imshow(ref_img_gradx)

    unsuploss = UnSupLoss(args)
    unsuploss(item['imgs_stage'], item['proj_matrices'])
    plt.show()




