# CSCF-Net

CASF-Net: Cross-attention And Cross-scale Fusion Network for Medical Image Segmentation (Submitted)

## CASF-Net Architecture
!(png)[https://github.com/ZhengJianwei2/CASF-Net/blob/main/main.png]

## 1. Environment setup

This code has been tested on on a personal laptop with Intel i7-10700H 3.8-GHz processor, 32-GB RAM, and an NVIDIA GTX3060t graphic card, Python 3.6, PyTorch 9.1, CUDA 10.0, cuDNN 7.6.  

## 2. Downloading necessary data:
* Kvasir-SEG Dataset:
[Segmented Polyp Dataset for Computer Aided Gastrointestinal Disease Detection](https://datasets.simula.no/kvasir-seg/) 

* ISIC 2018:
[ISIC 2018 Challenge](https://challenge.isic-archive.com/landing/2018/)

* GLAS,:
[Gland Segmentation Challenge Contest](https://warwick.ac.uk/fac/cross_fac/tia/data/glascontest/) 

and put them into data directory.

## 3. Download models

the models (loading models) and  he pretrained_models (loading model parameters):

* [models](https://drive.google.com/drive/folders/1GKnAeVtbn_PjnRURQlRqua6vWu-S8nZI)

and put them into models directory.

## 4.tran and test

    python  XX.py
           
The result will be saved in snapshots...

## 5. Cite Reference

Some of the codes in this repo are borrowed from:

[timm repo](https://github.com/rwightman/pytorch-image-models)

[PraNet repo](https://github.com/DengPingFan/PraNet)

[TransFuse repo](https://github.com/Rayicer/TransFuse)

## 6 Questions

Please drop an email to [hiderhao@gmail.com](hiderhao@gmail.com)
