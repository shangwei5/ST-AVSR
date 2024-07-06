# Arbitrary-Scale Video Super-Resolution with Structural and Textural Priors (ECCV2024)
---
[[arXiv]()] [[PDF]()]

This repository is the official PyTorch implementation of ST-AVSR: Arbitrary-Scale Video Super-Resolution with Structural and Textural Priors.

### Introduction


### Examples of the Demo
https://github.com/shangwei5/ST-AVSR/assets/43960503/3a8dd3c0-21fd-499c-8ccb-4362c6c5dcb0

https://github.com/shangwei5/ST-AVSR/assets/43960503/42babacd-1b23-480b-9984-c205c62f2b6d

https://github.com/shangwei5/ST-AVSR/assets/43960503/d18fd854-fee3-41f7-9c0a-de416ee49c8b

### Prerequisites
- Python >= 3.8, PyTorch == 1.13.0 (Perhaps >=1.8.0 is also OK), CUDA 11.7
- Requirements: opencv-python, numpy, matplotlib, imageio, scikit-image, thop, tqdm, cupy(cupy-cuda117), mmcv-full=1.6.2

### Datasets
Please download the RS-GOPRO datasets from [REDS](https://seungjunnah.github.io/Datasets/reds.html) (Type: Sharp) and [Vid4](https://drive.google.com/drive/folders/1An6hF1oYkeWxfOBxxKm073mvgIFrBNDA).

## Dataset Organization Form
```
|--REDS
    |--train
        |--train_sharp  
            |--video 1
                |--frame 1
                |--frame 2
                    ：
            |--video 2
                :
            |--video n
    |--val
        |--val_sharp
            |--video 1
                |--frame 1
                |--frame 2
                    ：
            |--video 2
             :
            |--video n
```
```
|--Vid4
    |--video 1
        |--frame 1
        |--frame 2
            ：
    |--video 2
        :
    |--video n
```

## Download Pre-trained Model
Please download the pre-trained model from [BaiduDisk](https://pan.baidu.com/s/1UBr9pQGhAHm66rr_VHzyTQ?pwd=47q3)(password:47q3). Please put the models to `./`.
Our results on REDS and Vid4 can also be downloaded from [BaiduDisk](https://pan.baidu.com/s/1WDO9wRFp5cA-dBSlKazcLg?pwd=rkf7)(password:rkf7) and [BaiduDisk](https://pan.baidu.com/s/1nqUUfEo6tFhiEZuY9bdYqw?pwd=6gv9)(password:6gv9).

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

