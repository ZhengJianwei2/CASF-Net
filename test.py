
import torch
import torch.nn.functional as F
import numpy as np
import os, argparse
from lib.TransFuse import TransFuse_S
from utils.data import test_dataset
import imageio
import cv2
import time
from thop import clever_format
from thop import profile
# from Exper.unet import ResNet34UnetPlus
# from Exper.unet import UNet
# from Exper.EnhancedUnet import EUNet


def mean_iou_np(y_true, y_pred, **kwargs):
    """
    compute mean iou for binary segmentation map via numpy
    """
    axes = (0, 1) 
    intersection = np.sum(np.abs(y_pred * y_true), axis=axes) 
    mask_sum = np.sum(np.abs(y_true), axis=axes) + np.sum(np.abs(y_pred), axis=axes)
    union = mask_sum  - intersection 
    
    smooth = 1
    iou = (intersection + smooth) / (union + smooth)
    return iou

def mean_recall_np(y_true, y_pred, **kwargs):
    """
    compute mean iou for binary segmentation map via numpy
    """
    axes = (0, 1) 
    intersection = np.sum(np.abs(y_pred * y_true), axis=axes) 
    mask_sum = np.sum(np.abs(y_true), axis=axes) + np.sum(np.abs(y_pred), axis=axes)
    union = mask_sum  - intersection 
    
    smooth = 1
    iou = (intersection + smooth) / (union + smooth)
    return iou

def mean_dice_np(y_true, y_pred, **kwargs):
    """
    compute mean dice for binary segmentation map via numpy
    """
    axes = (0, 1) # W,H axes of each image
    intersection = np.sum(np.abs(y_pred * y_true), axis=axes) 
    mask_sum = np.sum(np.abs(y_true), axis=axes) + np.sum(np.abs(y_pred), axis=axes)
    
    smooth = 1
    dice = 2*(intersection + smooth)/(mask_sum + smooth)
    return dice


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', type=str, default='././') 
    parser.add_argument('--data_path', type=str, default='././', help='path to test dataset')
    parser.add_argument('--save_path', type=str, default='././', help='path to save inference segmentation') 
    opt = parser.parse_args()
    
    time_start = time.time()
  
    model = TransFuse_S().cuda()


    model.load_state_dict(torch.load(opt.ckpt_path))
    model.cuda()
    model.eval()

    if opt.save_path is not None:
        os.makedirs(opt.save_path, exist_ok=True)

    print('evaluating model: ', opt.ckpt_path)

 

    image_root = '{}/images/'.format(opt.data_path)
    gt_root = '{}/masks/'.format(opt.data_path)
    num1 = len(os.listdir(gt_root))
    test_loader = test_dataset(image_root, gt_root,256)

    dice_bank = []
    iou_bank = []
    acc_bank = []


    for i in range(test_loader.size):
        image, gt, name = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()

        with torch.no_grad():
            p,q,r   = model(image)#P1,P2  = model(image)#

        res = F.upsample(p+q , size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()

    torch.cuda.synchronize()
    time_end = (time.time() - time_start)/test_loader.size

    print(time_end)
        # vis = (res - res.min()) / (res.max() - res.min() + 1e-8)
        # cv2.imwrite(opt.save_path+name, vis*255)

    res = 1*(res > 0.5)

    name=os.path.splitext(name)

    if opt.save_path is not None:
        imageio.imwrite(opt.save_path+'/pred/'+name[0]+'_pred.png', res)
        imageio.imwrite(opt.save_path+'/gt/'+name[0]+'_gt.png', gt)

    dice = mean_dice_np(gt, res)
    iou = mean_iou_np(gt, res)
    acc = np.sum(res == gt) / (res.shape[0]*res.shape[1])

    acc_bank.append(acc)
    dice_bank.append(dice)
    iou_bank.append(iou)

    print('Dice: {:.4f}, IoU: {:.4f}, Acc: {:.4f}'.
        format(np.mean(dice_bank), np.mean(iou_bank), np.mean(acc_bank)))
