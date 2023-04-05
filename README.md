# ViP3D: End-to-end Visual Trajectory Prediction via 3D Agent Queries (CVPR 2023)
### [Paper](https://arxiv.org/abs/2208.01582) | [Webpage](https://tsinghua-mars-lab.github.io/ViP3D/)
- This is the official repository of the paper: **ViP3D: End-to-end Visual Trajectory Prediction via 3D Agent Queries** (CVPR 2023).

[//]: # (## Getting Started)

[//]: # (- Installation)

[//]: # (- Prepare Dataset)

[//]: # (- Training and Evaluation)

##  Installation
#### Create conda environment
```bash
conda create -n vip3d python=3.6
```
#### Install pytorch
```bash
conda activate vip3d
pip install torch==1.10+cu111 torchvision==0.11.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
```
#### Install mmcv, mmdet
```bash
pip install mmcv-full==1.4.0 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.10/index.html
pip install mmdet==2.24.1
```
#### Install mmdet3d
```bash
cd ~
git clone https://github.com/open-mmlab/mmdetection3d.git
cd mmdetection3d
git checkout v0.17.1 # Other versions may not be compatible.
python setup.py install
pip install -r requirements.txt  # Install packages for mmdet3d
```

## Prepare Dataset
#### Download nuScenes full dataset (v1.0) and map expansion (v1.3) [here](https://www.nuscenes.org/download).
Only need to download Keyframe blobs and Radar blobs.


#### Structure
After downloading, the structure is as follows:
```
ViP3D
├── mmdet3d/
├── plugin/
├── tools/
├── data/
│   ├── nuscenes/
│   │   ├── maps/
│   │   ├── samples/
│   │   ├── v1.0-trainval/
│   │   ├── lidarseg/
```

#### Prepare data infos
Suppose data is saved at ```data/nuscenes/```.
```bash
python tools/data_converter/nusc_tracking.py
```

##  Training and Evaluation

#### Training
Train ViP3D using 3 historical frames and the ResNet50 backbone. It will load a pre-trained detector for weight initialization. Suppose the detector is at ```ckpts/detr3d_resnet50.pth```. It can be downloaded from [here](https://drive.google.com/file/d/1WHJYyg7RNcRj8_LfDnyNRZwxKKLeIZ9G/view?usp=sharing).
```bash
bash tools/dist_train.sh plugin/configs/vip3d_resnet50_3frame.py 8 --work-dir=work_dirs/vip3d_resnet50_3frame.1
```
The training stage requires ~ 17 GB GPU memory, and takes ~ 3 days for 24 epochs on 8× 3090 GPUS.

#### Evaluation
Coming soon! We are rewriting the evaluation metrics of trajectory prediction reference to the nuScenes official toolkit, making some improvements and making it easier to use for different models.


## License
The code and assets are under the Apache 2.0 license.

## Citation
If you find our work useful for your research, please consider citing the paper:
```bash
@article{vip3d,
  title={Vip3d: End-to-end visual trajectory prediction via 3d agent queries},
  author={Gu, Junru and Hu, Chenxu and Zhang, Tianyuan and Chen, Xuanyao and Wang, Yilun and Wang, Yue and Zhao, Hang},
  journal={arXiv preprint arXiv:2208.01582},
  year={2022}
}
```
