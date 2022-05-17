import os
from PIL import Image
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A
import random
import cv2


class SkinDataset(data.Dataset):
    """
    dataloader for skin lesion segmentation tasks
    """
    def __init__(self, image_root, gt_root):
        self.images = np.load(image_root)
        self.gts = np.load(gt_root)
        self.size = len(self.images)

        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
        self.gt_transform = transforms.Compose([
            transforms.ToTensor()])
        
        self.transform = A.Compose(
            [
                A.ShiftScaleRotate(shift_limit=0.15, scale_limit=0.15, rotate_limit=25, p=0.5, border_mode=0),
                A.ColorJitter(),
                A.HorizontalFlip(),
                A.VerticalFlip()
            ]
        )

    def __getitem__(self, index):
        
        image = self.images[index]
        gt = self.gts[index]
        gt = gt/255.0

        transformed = self.transform(image=image, mask=gt)
        image = self.img_transform(transformed['image'])
        gt = self.gt_transform(transformed['mask'])
        return image, gt

    def __len__(self):
        return self.size


def get_loader(image_root, gt_root, batchsize, shuffle=True, num_workers=4, pin_memory=True):

    dataset = SkinDataset(image_root, gt_root)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    return data_loader


class test_dataset:
    def __init__(self, image_root, gt_root):
        self.images = np.load(image_root)
        self.gts = np.load(gt_root)

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
            ])
        self.gt_transform = transforms.ToTensor()
        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image = self.images[self.index]
        image = self.transform(image).unsqueeze(0)
        gt = self.gts[self.index]
        gt = gt/255.0
        self.index += 1

        return image, gt



if __name__ == '__main__':
    path = '/home/hider/桌面/dataset/'
    tt = SkinDataset(path+'data_test.npy', path+'mask_test.npy')

    for i in range(5):
        img, gt = tt.__getitem__(i)
        print("img, gt:",img.shape,gt.shape)
        img = torch.transpose(img, 0, 1)
        img = torch.transpose(img, 1, 2)
        img = img.numpy()
        gt = gt.numpy()

        plt.imshow(img)
        plt.savefig('vis/'+str(i)+".jpg") 
 
        plt.imshow(gt[0])
        plt.savefig('vis/'+str(i)+'_gt.jpg')



# class PolypDataset(data.Dataset):
#     """
#     dataloader for polyp segmentation tasks
#     """
#     def __init__(self, image_root, gt_root, trainsize, augmentations):
#         self.trainsize = trainsize
#         self.augmentations = augmentations
#         print(self.augmentations)
#         self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
#         self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.png')]
#         self.images = sorted(self.images)
#         self.gts = sorted(self.gts)
#         self.filter_files()
#         self.size = len(self.images)
#         if self.augmentations == 'True':
#             print('Using RandomRotation, RandomFlip')
#             self.img_transform = transforms.Compose([
#                 transforms.RandomRotation(90, resample=False, expand=False, center=None, fill=None),
#                 transforms.RandomVerticalFlip(p=0.5),
#                 transforms.RandomHorizontalFlip(p=0.5),
#                 transforms.Resize((self.trainsize, self.trainsize)),
#                 transforms.ToTensor(),
#                 # transforms.RandomResizedCrop(224,scale=(0.5,1.0)),
#                 transforms.Normalize([0.485, 0.456, 0.406],
#                                      [0.229, 0.224, 0.225])])
#             self.gt_transform = transforms.Compose([
#                 transforms.RandomRotation(90, resample=False, expand=False, center=None, fill=None),
#                 transforms.RandomVerticalFlip(p=0.5),
#                 transforms.RandomHorizontalFlip(p=0.5),
#                 transforms.Resize((self.trainsize, self.trainsize)),
#                 transforms.ToTensor()])
            
#         else:
#             print('no augmentation')
#             self.img_transform = transforms.Compose([
#                 transforms.Resize((self.trainsize, self.trainsize)),
#                 transforms.ToTensor(),
#                 # transforms.RandomResizedCrop(224,scale=(0.5,1.0)),
#                 transforms.Normalize([0.485, 0.456, 0.406],
#                                      [0.229, 0.224, 0.225])])
            
#             self.gt_transform = transforms.Compose([
#                 transforms.Resize((self.trainsize, self.trainsize)),
#                 transforms.ToTensor()])
            

#     def __getitem__(self, index):
        
#         image = self.rgb_loader(self.images[index])
#         gt = self.binary_loader(self.gts[index])
        
#         seed = np.random.randint(2147483647) # make a seed with numpy generator 
#         random.seed(seed) # apply this seed to img tranfsorms
#         torch.manual_seed(seed) # needed for torchvision 0.7
#         if self.img_transform is not None:
#             image = self.img_transform(image)
            
#         random.seed(seed) # apply this seed to img tranfsorms
#         torch.manual_seed(seed) # needed for torchvision 0.7
#         if self.gt_transform is not None:
#             gt = self.gt_transform(gt)
#         return image, gt

#     def filter_files(self):
#         assert len(self.images) == len(self.gts)
#         images = []
#         gts = []
#         for img_path, gt_path in zip(self.images, self.gts):
#             img = Image.open(img_path)
#             gt = Image.open(gt_path)
#             if img.size == gt.size:
#                 images.append(img_path)
#                 gts.append(gt_path)
#         self.images = images
#         self.gts = gts

#     def rgb_loader(self, path):
#         with open(path, 'rb') as f:
#             img = Image.open(f)
#             return img.convert('RGB')

#     def binary_loader(self, path):
#         with open(path, 'rb') as f:
#             img = Image.open(f)
#             # return img.convert('1')
#             return img.convert('L')

#     def resize(self, img, gt):
#         assert img.size == gt.size
#         w, h = img.size
#         if h < self.trainsize or w < self.trainsize:
#             h = max(h, self.trainsize)
#             w = max(w, self.trainsize)
#             return img.resize((w, h), Image.BILINEAR), gt.resize((w, h), Image.NEAREST)
#         else:
#             return img, gt

#     def __len__(self):
#         return self.size




# def get_loader_polyp(image_root, gt_root, batchsize, trainsize, shuffle=True, num_workers=4, pin_memory=True, augmentation=False):

#     dataset = PolypDataset(image_root, gt_root, trainsize, augmentation)
#     data_loader = data.DataLoader(dataset=dataset,
#                                   batch_size=batchsize,
#                                   shuffle=shuffle,
#                                   num_workers=num_workers,
#                                   pin_memory=pin_memory)
#     return data_loader


# class test_dataset_p:
#     def __init__(self, image_root, gt_root, testsize):
#         self.testsize = testsize
#         self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
#         self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.tif') or f.endswith('.png')]
#         self.images = sorted(self.images)
#         self.gts = sorted(self.gts)
#         self.transform = transforms.Compose([
#             transforms.Resize((self.testsize, self.testsize)),
#             transforms.ToTensor(),
#             transforms.Normalize([0.485, 0.456, 0.406],
#                                  [0.229, 0.224, 0.225])])
#         self.gt_transform = transforms.ToTensor()
#         self.size = len(self.images)
#         self.index = 0

#     def load_data(self):
#         image = self.rgb_loader(self.images[self.index])
#         image = self.transform(image).unsqueeze(0)
#         gt = self.binary_loader(self.gts[self.index])
#         # print("load:",image.shape,gt.shape)
#         name = self.images[self.index].split('/')[-1]
#         if name.endswith('.jpg'):
#             name = name.split('.jpg')[0] + '.png'
#         self.index += 1
#         return image, gt, name

#     def rgb_loader(self, path):
#         with open(path, 'rb') as f:
#             img = Image.open(f)
#             return img.convert('RGB')

#     def binary_loader(self, path):
#         with open(path, 'rb') as f:
#             img = Image.open(f)
#             return img.convert('L')



# if __name__ == '__main__':
#     path = 'data/'
#     tt = SkinDataset(path+'data_train.npy', path+'mask_train.npy')

#     for i in range(50):
#         img, gt = tt.__getitem__(i)

#         img = torch.transpose(img, 0, 1)
#         img = torch.transpose(img, 1, 2)
#         img = img.numpy()

#         gt = gt.numpy()

#         plt.imshow(img)
#         plt.savefig('vis/'+str(i)+".jpg")
 
#         plt.imshow(gt[0])
#         plt.savefig('vis/'+str(i)+'_gt.jpg')
