# Project-Building-heatloss-analysis

# 환경설정

## 1. 가상환경(kict_demo), 파이썬(3.8.5), 아나콘다 설치
```
conda create -n kict_demo python=3.8.5 anaconda

conda activate kict_demo
```

## 2. GPU(CUDA 10.1)에 해당하는 Pytorch(1.8.0) 설치
```
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=10.1 -c pytorch
```
#### * 원래 pytorch 1.4.0이지만 DexiNed추가로 1.8.0으로 변경

#### * CUDA, GPU 사용 가능 여부 확인
```
import torch
USE_CUDA = torch.cuda.is_available()
print(USE_CUDA)
device = torch.device('cuda:0' if USE_CUDA else 'cpu')
print('학습을 진행하는 기기:',device)
```

## 3. 가상환경 활용을 위한 ipykernel 및 필요 module import

#### * ipykernel 설치

```
pip install ipykernel
python -m ipykernel install --user --name kict_demo --display-name "kict_demo"
```
#### * module import
```
conda install selenium
pip install kornia
pip install opencv-python
```

## 999. 가상환경 삭제 (혹시모를..)
```
jupyter kernelspec uninstall kict_demo
conda env remove -n kict_demo
```
