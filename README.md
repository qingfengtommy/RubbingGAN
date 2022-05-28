# RubbingGAN

RubbingGAN - Pytorch Implementation (AI for Content Creation Workshop CVPR 2022)
<a href=https://doi.org/10.48550/arXiv.2205.03743><img src="https://img.shields.io/badge/arXiv-2205.03743-b31b1b.svg" alt="ci"></a>

# Introduction

This is the implementation of the paper "End-to-End Rubbing Restoration Using Generative Adversarial Network. Our goal is to restore the imcomplete calligraphy characters on the traidional Chinese stone inscriptions (Rubbings).

# Dataset

We collect the dataset of ZhangMenglongBei, which used for rubbing restoration.

We have the following datasets: 

- Training Dataset

- Testing Dataset

- Restoration (Incomplete) Dataset
  
<img src="https://github.com/qingfengtommy/RubbingGAN/blob/main/imgs/Fig3a.png" alt="Training Dataset" align="left" width="400"/>

<img src="https://github.com/qingfengtommy/RubbingGAN/blob/main/imgs/Fig3b.png" alt="Restoration Dataset" width="400"/>
 
# Train

``` python RubbingGAN.py --dataroot PATH TO Training Dataset --valDataroot PATH TO Validation Dataset --exp PATH TO store outputs of generating images --log PATH TO store logs ```

# Restoration Result

<img src=https://github.com/qingfengtommy/RubbingGAN/blob/main/imgs/Fig11.png alt="Resotration Result" width="500"/>

# Reference

- pix2pix-pytorch https://github.com/taey16/pix2pixBEGAN.pytorch

- began—pytorch https://github.com/taey16/pix2pixBEGAN.pytorch
