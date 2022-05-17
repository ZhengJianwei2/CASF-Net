import torch
from torch.autograd import Variable
import argparse
from datetime import datetime
from lib.TransFuse import TransFuse_S as TransModel
from utils.data import get_loader, test_dataset
from utils.utils import adjust_lr, AvgMeter
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from test_isic import mean_dice_np, mean_iou_np
import logging
import os


def structure_loss(pred, mask):
    weit = 1 + 5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    wbce = (weit*wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask)*weit).sum(dim=(2, 3))
    union = ((pred + mask)*weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1)/(union - inter+1)
    return (wbce + wiou).mean()


def train(train_loader, model, optimizer, epoch, best_loss):
    model.train()
    loss_record1,loss_record2, loss_record3, loss_record4 = AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter()
    accum = 0
    for i, pack in enumerate(train_loader, start=1):
        # ---- data prepare ----
        images, gts = pack
        images = Variable(images).cuda()
        gts = Variable(gts).cuda()

        # ---- forward ----
        P1, P2,P3= model(images)
        # ---- loss function ----
        loss_P1 = structure_loss(P1, gts)
        loss_P2 = structure_loss(P2, gts)
        loss_P3 = structure_loss(P3, gts)
        # loss_P4 = structure_loss(P4, gts)
        loss = 0.25*loss_P1 + 0.25*loss_P2 + 0.5*loss_P3 #+ 0.4*loss_P4  

        # ---- backward ----
        loss.backward() 
        torch.nn.utils.clip_grad_norm_(model.parameters(), opt.grad_norm)
        optimizer.step()
        optimizer.zero_grad()

        # ---- recording loss ----
        loss_record1.update(loss_P1.data, opt.batchsize)
        loss_record2.update(loss_P2.data, opt.batchsize)
        loss_record3.update(loss_P3.data, opt.batchsize)
        # loss_record4.update(loss_P4.data, opt.batchsize)

        # ---- train visualization ----
        if i % 350 == 0 or i == total_step:
            print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], '
                  '[lateral-2: {:.4f}, lateral-3: {:0.4f}, lateral-4: {:0.4f}]'.  
                  format(datetime.now(), epoch, opt.epoch, i, total_step,
                         loss_record1.show(), loss_record2.show(), loss_record3.show()))

    save_path = 'snapshots/{}/'.format(opt.train_save)
    os.makedirs(save_path, exist_ok=True)
    if (epoch+1) % 1 == 0:
        meanloss = test(model, opt.test_path)
        if meanloss < best_loss:
            print('new best loss: ', meanloss)
            best_loss = meanloss
            torch.save(model.state_dict(), save_path + 'TransFuse-%d.pth' % epoch)
            print('[Saving Snapshot:]', save_path + 'TransFuse-%d.pth'% epoch)
    return best_loss


def test(model, path, dataset):

    model.eval()
    data_path = os.path.join(path, dataset)
    image_root = '{}/images/'.format(data_path)
    gt_root = '{}/masks/'.format(data_path)
    test_loader = test_dataset(image_root, gt_root, opt.trainsize)
    num1 = len(os.listdir(gt_root))
    DSC = 0.0

    for i in range(test_loader.size):
        image, gt,_ = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()
        
        with torch.no_grad():
            res  = model(image)

        res = F.upsample( res, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        input = res
        target = np.array(gt)
        N = gt.shape
        smooth = 1
        input_flat = np.reshape(input, (-1))
        target_flat = np.reshape(target, (-1))
        intersection = (input_flat * target_flat)
        dice = (2 * intersection.sum() + smooth) / (input.sum() + target.sum() + smooth)
        dice = '{:.4f}'.format(dice)
        dice = float(dice)
        DSC = DSC + dice
    
    # print("meandice:",DSC / num1)
    return DSC / num1


if __name__ == '__main__':
    dict_plot = {'CVC-300':[], 'CVC-ClinicDB':[], 'Kvasir':[], 'CVC-ColonDB':[], 'ETIS-LaribPolypDB':[], 'test':[]}
    name = ['CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB', 'test']
    ##################model_name#############################
    #########################################################
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=30, help='epoch number')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--batchsize', type=int, default=2, help='training batch size')
    parser.add_argument('--grad_norm', type=float, default=2.0, help='gradient clipping norm')
    parser.add_argument('--trainsize', type=int,default=256, help='training dataset size')
    parser.add_argument('--train_path', type=str,default='./data/ClinicDB/train/', help='path to train dataset')
    parser.add_argument('--test_path', type=str,default='./data/ClinicDB/', help='path to test dataset')
    parser.add_argument('--train_save', type=str, default='refer')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 of adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 of adam optimizer')
    parser.add_argument('--optimizer', type=str,default='AdamW', help='choosing optimizer AdamW or SGD')

    opt = parser.parse_args()
    logging.basicConfig(filename='train_log.log',
                        format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                        level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')
    # ---- build models ----
    model = TransModel(pretrained=True).cuda()
    params = model.parameters()
    if opt.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(params, opt.lr, weight_decay=1e-4)
    else:
        optimizer = torch.optim.SGD(params, opt.lr, weight_decay=1e-4, momentum=0.9)
    # optimizer = torch.optim.Adam(params, opt.lr, betas=(opt.beta1, opt.beta2))
     
    image_root = '{}/images/'.format(opt.train_path)
    gt_root = '{}/masks/'.format(opt.train_path)

    train_loader = get_loader(image_root, gt_root, batchsize=opt.batchsize, trainsize=opt.trainsize, augmentation = False)
    total_step = len(train_loader)

    print("#"*20, "Start Training", "#"*20)

    best_loss = 1e5
    for epoch in range(1, opt.epoch + 1):
        adjust_lr(optimizer, opt.lr, epoch, 0.1, 200)
        best_loss = train(train_loader, model, optimizer, epoch, opt.test_path)