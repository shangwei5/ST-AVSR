# Arbitrary-Scale Video Super-Resolution with Structural and Textural Priors (ECCV2024)
---
[[arXiv]()] [[PDF]()]

This repository is the official PyTorch implementation of ST-AVSR: Arbitrary-Scale Video Super-Resolution with Structural and Textural Priors.

### Introduction


### Examples of the Demo

### Prerequisites
- Python >= 3.8, PyTorch == 1.13.0 (Perhaps >=1.8.0 is also OK), CUDA 11.7
- Requirements: opencv-python, numpy, matplotlib, imageio, scikit-image, thop, tqdm, cupy(cupy-cuda117), mmcv-full=1.6.2

### Datasets
Please download the RS-GOPRO datasets from [REDS]() and [Vid4]()

## Dataset Organization Form
```
|--dataset
    |--train  
        |--video 1
            |--GS
                |--frame 1
                |--frame 2
                    ：
            |--RS
                |--frame 1
                |--frame 2
                    ： 
        |--video 2
            :
        |--video n
    |--valid
        |--video 1
            |--GS
                |--frame 1
                |--frame 2
                    ：
            |--RS
                |--frame 1
                |--frame 2
                    ：   
        |--video 2
         :
        |--video n
    |--test
        |--video 1
            |--GS
                |--frame 1
                |--frame 2
                    ：
            |--RS
                |--frame 1
                |--frame 2
                    ：   
        |--video 2
         :
        |--video n
```

## Download Pre-trained Model
Please download the pre-trained model from [BaiduDisk]()(password:pjdx). Please put the models to `./`.

## Getting Started
### 1) Testing
1.Testing on RS-GOPRO dataset:
```
python -m torch.distributed.launch --nproc_per_node=1 --master_port=1234 main_test_srsc_rsflow_multi_distillv2.py --opt options/test_srsc_rsflow_multi_distillv2_psnr.json  --dist True
```
Please change `data_root` and `pretrained_netG` in options according to yours.

1.Testing on real RS data:
```
python -m torch.distributed.launch --nproc_per_node=1 --master_port=1234 main_test_srsc_rsflow_multi_distillv2_real.py --opt options/test_srsc_rsflow_multi_distillv2_real.json  --dist True
```
Please change `data_root` and `pretrained_netG` in options according to yours.

### 2) Training
1.Training the first stage: (8 GPUs)
```
python -m torch.distributed.launch --nproc_per_node=8 --master_port=1234 main_train_srsc_rsflow_multi.py --opt options/train_srsc_rsflow_multi_psnr.json --dist True
```
Please change `data_root` and `pretrained_rsg` in options according to yours.


2.Training the second stage (adding self-distillation): (8 GPUs)
```
python -m torch.distributed.launch --nproc_per_node=8 --master_port=1234 main_train_srsc_rsflow_multi_distillv2.py --opt options/train_srsc_rsflow_multi_distillv2_psnr.json  --dist True
```
Please change `data_root`, `pretrained_rsg` and `pretrained_netG` in options according to yours.

## Cite
If you use any part of our code, or ST-AVSR is useful for your research, please consider citing:
```

```

## Contact
If you have any questions, please contact csweishang@gmail.com.

