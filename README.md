# Geospatial Foundation Models 

Towards Geospatial Foundation Models via Continual Pretraining: [\[arxiv\]](https://arxiv.org/abs/2302.04476)

<!-- <p><img src="figures/gfm.png" width="1000" /></p> -->
<div align="center">
    <img src="gfm.png" height="250px" />
</div>
<!-- <img src="gfm.png" width="300"> -->

Bridging Remote Sensors with Multisensor Geospatial Foundation Models: [\[arxiv\]](https://arxiv.org/abs/2404.01260)


## Setup
First follow the instructions for the SimMIM repo installation [here](https://github.com/microsoft/SimMIM#installation).
Then, within your newly created virtual environment, run
```bash
pip install torchgeo
pip install opencv-python
```

## GeoPile and GFM Pretrained Model
**GFM**: The GeoPile and GFM pretrained model are avaliable on [OneDrive](https://1drv.ms/f/s!AkTn76m907OThpRJjH8ehfskbgCLXw?e=ZJreFo). As the GeoPile is a collection of data from various sources, please be sure to cite the original data sources (references [9, 29, 33, 35, 48] in the paper) as well if you use this in future research.

**msGFM**: To be releasesd

## Pretraining

**GFM**:

To conduct your own pretraining, first download the GeoPile dataset and unzip it on your system.
Also, download the ImageNet-22k pretrained model from the SimMIM repo
```
wget https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224_22k.pth
```
and place it under the following folder structure.
```
output
   |- simmim_finetune
      |- swin_base_patch4_window7_224_22k.pth
```
The basic command for pretraining is as follows:
```bash
python -m torch.distributed.launch --nproc_per_node 8 main_teacher.py \
--cfg configs/simmim_pretrain__swin_base__img192_window6__100ep.yaml --batch-size 128 \
--data-path /path/to/GeoPileV0/ --tag gfm --pretrained output/simmim_finetune/swin_base_patch4_window7_224_22k.pth
```

**msGFM**: To be releasesd

## Finetuning
To perform finetuning, place the GFM pretrained model in the following folder structure.
```
output
   |- simmim_pretrain
      |- gfm.pth
```
An example command for finetuning is as follows:
```bash
python -m torch.distributed.launch --nproc_per_node 4 main_finetune.py --cfg configs/BEN.yaml --batch-size 128 \
--data-path /path/to/bigearthnet/ --pretrained output/simmim_pretrain/gfm.pth --tag BEN --train_frac 0.01
```
## Citation
```
@inproceedings{mendieta2023towards,
  title={Towards Geospatial Foundation Models via Continual Pretraining},
  author={Mendieta, Mat{\'\i}as and Han, Boran and Shi, Xingjian and Zhu, Yi and Chen, Chen},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={16806--16816},
  year={2023}
}
```
```
@misc{han2024bridging,
      title={Bridging Remote Sensors with Multisensor Geospatial Foundation Models}, 
      author={Boran Han and Shuai Zhang and Xingjian Shi and Markus Reichstein},
      year={2024},
      eprint={2404.01260},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
## Acknowledgement
This code is based on [SimMIM](https://github.com/microsoft/SimMIM).
