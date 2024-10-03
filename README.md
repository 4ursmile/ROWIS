# ROWIS
---
> This is official implementation of the paper "Towards Real-Time Open World Instance Segmentation".
---
## Abstract
	
Instance segmentation is a common task in computer vision
specifically, and computer science in general. Its applications are widely
used in areas such as autonomous driving and automotive systems. However, current instance segmentation models are often limited, as they only
perform well on fixed training sets. This creates a significant challenge in
real-world applications, where the number of classes is strongly dependent on the training data. To address this limitation, we propose the concept of Open World Instance Segmentation (OWIS) with two main objectives: (1) segmenting instances not present in the training set as an “unknown" class, and (2) enabling models to incrementally learn new classes
without forgetting previously learned ones, with minimal cost and effort.
These objectives are derived from open world object detection task Joseph et al.
We also introduce new datasets following a novel protocol for evaluation,
along with a strong baseline method called ROWIS (Real-Time Open
World Instance Segmentor), which incorporates an advanced energy-based strategy for unknown class identification. Our evaluation, based
on the proposed protocol, demonstrates the effectiveness of ROWIS in
addressing real-world challenges. his research will encourage further exploration of the OWIS problem and contribute to its practical adoption.

## Installation

This project is based on [CVPR 2022] [SparseInst](https://github.com/hustvl/SparseInst): Sparse Instance Activation for Real-Time Instance Segmentation, which based on [Detectron2](https://github.com/facebookresearch/detectron2). Please follow the installation instructions of Detectron2 to install the required dependencies.
or you can use the following command to install the required dependencies:

```bash
pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

Then clone this repository:

```bash
git clone -b main-train --single-branch https://github.com/4ursmile/ROWIS
```
Now prepare the pre-trained backbone weights:

```bash
mkdir trained
mkdir pretrained_models

wget https://dl.fbaipublicfiles.com/detectron2/ImageNetPretrained/MSRA/R-50.pkl -P pretrained_models/

wget https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/resnet50d_ra2-464e36ba.pth -P pretrained_models/
```

This is some missing libraries that you may need to install:

```bash
pip install scipy opencv-python timm shapely gdown
```

## Datasets
Firstly, you need to download the coco images and our annotations.

Run the following in python environment:

```python
import os
import gdown
import zipfile


if not os.path.exists('datasets'):
    os.makedirs('datasets')
if not os.path.exists('datasets/OWIS'):
    os.makedirs('datasets/OWIS')
if not os.path.exists('datasets/OWIS/annotations'):
    os.makedirs('datasets/OWIS/annotations')
file_id = "1yLG9NDvAfIh2BVGiOyzcXm4BMPwOY7K7" 
url = f"https://drive.google.com/uc?id={file_id}"
output = "datasets/OWIS/annotations/dataset.zip"
gdown.download(url, output, quiet=False)

with zipfile.ZipFile(output, 'r') as zip_ref:
    zip_ref.extractall('datasets/OWIS/annotations')
```
  
Then download the COCO images:
  
```bash
# create folder
mkdir datasets/coco2017
cd datasets/coco2017
# download images
wget -c http://images.cocodataset.org/zips/val2017.zip
wget -c http://images.cocodataset.org/zips/train2017.zip

# unzip images
unzip -q val2017.zip
unzip -q train2017.zip

# remove zip files
rm val2017.zip
rm train2017.zip

# return to the main folder
cd ../..
```
The dataset folder should look like this:
```
datasets
│-- coco2017
│   │-- train2017
│   │   │-- 000000000009.jpg
│   │   │-- ...
│   │-- val2017
│   │   │-- 000000000139.jpg
│   │   │-- ...
│-- OWIS
│   │-- annotations
│   │   │-- dataset
│   │   │   │-- annotations
│   │   │   │   │-- T1_instance_train_new.json
│   │   │   │   │-- T1_instance_val_new.json
│   │   │   │-- |-- ..
```

That's it! You are ready to train the model.

## Training
Training and evaluation in one command:

```bash
# provide excutable permission to the script
chmod +x run.sh
# run the script
./run.sh
```
That's all! The model will be trained and evaluated on the validation set.



