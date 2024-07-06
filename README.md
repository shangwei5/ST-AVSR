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
1. Processing the entire video frame:
```
bash test_sequence.sh
```
Please change `--data_path` according to yours.

2. Processing frame by frame :
```
bash test.sh
```
Please change `--data_path` according to yours.

### 2) Training

We use an NVIDIA RTX A6000 (48GB) for training. Please adjust the `batch_size` and `test{'n_seq'}` in options based on your GPU memory.
```
python -m torch.distributed.launch --nproc_per_node=1 --master_port=1234 train.py --opt options/train_refsrrnn_cuf_siren_adists_only_future_t2.json --dist True
```
Please change `gpu_ids`, `path{'root', 'images'}`, and `data_root` in options according to yours.


## Cite
If you use any part of our code, or ST-AVSR is useful for your research, please consider citing:
```

```

## Contact
If you have any questions, please contact csweishang@gmail.com.

