co-regularied-alignment-for-domain-adaptation
===========================
This is an UNOFFICIAL implemnetation of [Co-regularized Alignment for Unsupervised Domain
Adaptation](https://arxiv.org/pdf/1811.05443.pdf)

****

## Prerequisite
* python 3.6
* PyTorch 1.1
* tensorboardX
* numpy
* cv2
* tqdm

***
## Training losses & results from MNIST->SVHN task
|agreement loss|diverse loss|
|---|---
|<img src="./images/agree_loss.png" width="320" height="240">|<img src="./images/div_loss.png" width="320" height="240">
***

<img src="./images/loss1_1.png" width="600" height="240">
<img src="./images/loss1_2.png" width="600" height="240">

***

|net_num|validation|test|
|---|---|---
|net1|<img src="./images/val1.png" width="320" height="240">|<img src="./images/test1.png" width="320" height="240">
|net2|<img src="./images/val2.png" width="320" height="240">|<img src="./images/test2.png" width="320" height="240">

***
TODO:
1. DIRT-T model

***
Note: 
See https://github.com/lyakaap/VAT-pytorch for original implementation of VAT loss
