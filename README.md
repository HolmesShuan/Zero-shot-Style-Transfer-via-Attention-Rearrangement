> [!NOTE]
> Apologies for the delay in code submission due to other ongoing commitments. We have created this repository to ensure that interested readers can locate the official source for our work. Thank you for your understanding and stay tuned for updates.

## Project Name
Z-STAR: Zero-shot Style Transfer via Attention Rearrangement (Reweighting)
<img width="1154" alt="image" src="https://github.com/user-attachments/assets/4bf65ab7-a5d3-4400-b8aa-83ebca942e87">


## Introduction
Z-STAR is an innovative zero-shot (training-free) style transfer method that leverages the generative prior knowledge within a pre-trained diffusion model. By employing an attention rearrangement strategy, it effectively fuses content and style information without the need for retraining or tuning for each input style.

## Main Contributions
- A zero-shot image style transfer method that does not require retraining or tuning.
- A rearranged attention mechanism for disentangling and fusing content/style information in the diffusion latent space.
  
## Installation
### CUDA Version
```shell
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2021 NVIDIA Corporation
Built on Wed_Jul_14_19:41:19_PDT_2021
Cuda compilation tools, release 11.4, V11.4.100
Build cuda_11.4.r11.4/compiler.30188945_0
```
### Python Version
```shell
Python 3.8.8 (default, Jul 20 2021, 08:48:08) 
[GCC 7.5.0] on linux
```
### Other Packages
```shell
# Please refer to the requirements.txt
pip3 install -r requirements.txt
```
### Download SD1.5 Model
```shell
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt-get install git-lfs
git lfs install
git lfs clone https://huggingface.co/runwayml/stable-diffusion-v1-5
# or
# git lfs clone https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5
# For Chinese users, please refer to the following website https://blog.csdn.net/BetrayFree/article/details/134023877.
```

## Usage
```shell
python3.8 demo.py # the results can be found in 'workdir/demo'
# python3.8 demo.py --content_img_folder <path_to_content_image> --style_img_folder <path_to_style_image> --sub_exp_name <path_to_save_stylized_image>
```

## Paper and Code
This project is the official implementation of the paper "Z∗: Zero-shot Style Transfer via Attention Rearrangement," for details, please refer to [the paper](https://arxiv.org/abs/2311.16491).

## License
This project is under the **Apache license 2.0** license. Please feel free to incorporate it into your project.

## Acknowledgement
This repository was developed based on the open-source codes from [MasaCtrl](https://github.com/TencentARC/MasaCtrl) and [null-text-inversion](https://null-text-inversion.github.io/). We would like to express our gratitude to the authors for their contributions to the community through their open-source efforts.
