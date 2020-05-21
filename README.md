# Predict & Cluster

## Introduction
This repository contains the code for the paper "PREDICT & CLUSTER: Unsupervised Skeleton Based Action Recognition", which is available [here](https://arxiv.org/abs/1911.12409). To appear in CVPR 2020. 
The unsupervised approach implements two strategies discussed in the paper, Fixed-state(FS) and Fixed-weight(FW). It is tested on NW-UCLA and NTU-RGBD(60) dataset.


[![Alt text](https://img.youtube.com/vi/-dcCFUBRmwE/0.jpg)](https://www.youtube.com/watch?v=-dcCFUBRmwE)

## Abstract
We propose a novel system for unsupervised skeleton-based action recognition. Given inputs of body-keypoints sequences obtained during various movements, 
our system associates the sequences with actions. Our system is based on an encoder-decoder recurrent neural network, where the encoder learns a separable feature 
representation within its hidden states formed by training the model to perform the prediction task. We show that according to such unsupervised training, the decoder 
and the encoder self-organize their hidden states into a feature space which clusters similar movements into the same cluster and distinct movements into distant clusters.

## Examples
The python notebook ucla_demo.ipynb is the demonstration notebook for FS and FW strategies on NW-UCLA dataset, which is available on http://wangjiangb.github.io/my_data.html. The preprocessed UCLA data is included in ucla_data directory, please refer to ucla_demo.ipynb for more info. The data preprocessing part for UCLA dataset is also incorporated within the notebook, and is ready to run. The results shown in the notebook is not the most optimal one as we got in the paper, but is almost comparable.

Other python scripts are extracted from the notebooks we wrote for NTU-RGBD 60 datasets, but not tested yet.

## Requirements
1. Tensorflow 1.14.0
2. Python 3
3. scikit-learn 0.21.2
4. matplotlib 3.1.0
5. numpy 1.16.4

## Citation

Please cite the [PREDICT & CLUSTER: Unsupervised Skeleton Based Action Recognition (Arxiv; To appear in CVPR 2020)](https://arxiv.org/abs/1911.12409) if you use this code:
```
@inproceedings{su2020predict,
  title={PREDICT \& CLUSTER: Unsupervised Skeleton Based Action Recognition},
  author={Su, Kun and Liu, Xiulong and Shlizerman, Eli}
  journal={CVPR, IEEE Computer Society Conference on Computer Vision and Pattern Recognition}
  year={2020}
}
```
