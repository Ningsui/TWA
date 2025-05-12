This is the code for the TGRS 2024 paper "Two-Way Assistant: A Knowledge Distillation Object Detection Method for Remote Sensing Images"

Our codes are based on [MMDetection](https://github.com/open-mmlab/mmdetection). Please follow the installation of MMDetection 
and make sure you can run it successfully.

Add and Replace the codes
Add the configs/. in our codes to the configs/ in mmdetectin's codes.
Add the mmdet/models/detectors/. in our codes to the mmdet/models/detectors/.

Unzip LEVIR and SSDD dataset into data/coco/

## Citation
If you find our repo useful for your research, please cite us:
```
@ARTICLE{10453608,
  author={Yang, Xi and Zhang, Sheng and Yang, Weichao},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={Two-Way Assistant: A Knowledge Distillation Object Detection Method for Remote Sensing Images}, 
  year={2024},
  volume={62},
  number={},
  pages={1-10},
  keywords={Feature extraction;Remote sensing;Detectors;Image coding;Object detection;Head;Computational modeling;Knowledge distillation (KD);object detection;remote sensing images;two-way assistant (TWA)},
  doi={10.1109/TGRS.2024.3371681}}
```
