# Learning-Constrained-Structured-Spaces-with-Application-to-Multi-Graph-Matching
This is the official code for the AISTATS 2023 paper "Learning Constrained Structured Spaces with Application to Multi-Graph Matching".

Our method for end-to-end learning multi-graph matching accounts for pairwise missing correspondences and allows for minimizing the structured loss without relaxing the matching prediction. We extend the direct loss minimization to settings in which the black-box solvers are computationally inefficient, as in the setting of multi-graph matchings. Thus, our method allows learning cycle-consistent matchings, while not using a cycle-consistent matching solver, while theoretically recovering the constrained multi-graph matching optimal solution.

This code demonstrates the effectiveness of our method in balanced and unbalanced two-graph, hypergraph, and multi-graph matching tasks.

## Cycle-consistency loss
<p align="center">
  <img src="https://user-images.githubusercontent.com/46455293/217747708-2454dc59-e18f-4364-b15d-4a7b1f73663c.svg" width="450" title="cycle_consistency_loss">
</p>


A toy illustration of our cycle-consistency loss for 3-graph $G_i, G_k, G_j$ unbalanced (partial) matching. Pairwise matchings $y^{\ast}(x^{ij}),y^{\ast}(x^{ik}),y^{\ast}(x^{jk})$ are predicted. 
The cycle-consistency constraint on the matching between $G_i$ and $G_k$ w.r.t. passing the matching through image $G^j$ involves inequalities of the form $y^{\ast}(x^{ij})y^{\ast}(x^{jk}) \le y^{\ast}(x^{ik})$. This constraint translates to the penalty function $max(0, y^{\ast}(x^{ij})y^{\ast}(x^{jk}) -y^{\ast}(x^{ik}))$. Depicted is the cycle-consistency loss of the predicted pairwise matching $y^{\ast}(x^{ik})$. 


# How to run this code

This code and setup is based on the well-organized [ThinkMatch](https://github.com/Thinklab-SJTU/ThinkMatch) project. We'd like to thank its creators for making this  unified matching implementations code base publicly available.

## Prepare the datasets

1. PascalVOC-Keypoint
    1. Download [VOC2011 dataset](http://host.robots.ox.ac.uk/pascal/VOC/voc2011/index.html) and make sure it looks like ``data/PascalVOC/VOC2011``
    1. Download keypoint annotation for VOC2011 from [Berkeley server](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/shape/poselets/voc2011_keypoints_Feb2012.tgz) or [google drive](https://drive.google.com/open?id=1D5o8rmnY1-DaDrgAXSygnflX5c-JyUWR) and make sure it looks like ``data/PascalVOC/annotations``
    1. The train/test split is available in ``data/PascalVOC/voc2011_pairs.npz``

1. Willow-Object-Class
    1. Download [Willow-ObjectClass dataset](http://www.di.ens.fr/willow/research/graphlearning/WILLOW-ObjectClass_dataset.zip)
    1. Unzip the dataset and make sure it looks like ``data/WILLOW-ObjectClass``

## Requirements
Please see requirements file. Our code is developed and tested on Python 3.8.3, torch==1.9.0, torch-geometric==1.7.2, torch-scatter==2.0.7, torch-sparse==0.6.10, torch-spline-conv==1.2.1, CUDA Version: 11.4

Also, install and build LPMP (You may need gcc-9 to successfully build LPMP)
```bash 
python -m pip install git+https://git@github.com/rogerwwww/lpmp.git 
```

## Running the code 
```bash
python train_eval.py --cfg path/to/yaml
```

and replace ``path/to/your/yaml`` by path to your configuration file, e.g.
```bash
python train_eval.py --cfg experiments/vvgg16_ngmv2_voc.yaml
```

Matching on the Willow-Object-Class dataset is balanced. Meaning, any sampled pair of images from the same class will consist of the same 10 semantic keypoints.
In yaml config file, this is controlled by the 'MATCHING_TYPE: Balanced' argument.

Matching on the PascalVOC-Keypoint dataset is unbalanced. Meaning, a sampled pair of images from the same class may consist of both outliers in source and in target image. We do not filter keypoint to circumvent this natural imbalance. 
In yaml config file, this is controlled by the 'MATCHING_TYPE: Unalanced', 'filter_type: NoFilter' arguments.
In config file we set 'PROBLEM.SRC_OUTLIER = True', and 'PROBLEM.TGT_OUTLIER = True'.

## Other settings:

'samples_per_num_train' controls how many perturbations will be conducted for each permutation representation. We set samples_per_num_train=38 in our experiments.

'loss_epsilon' controls the loss-augmentation magnitude of the direct loss minimization gradient. See Appendix A.2 for some insight regarding this parameter.  In general, we set loss_epsilon to a small value and increase it by a certain percentage whenever the loss is positive and the gradients are zero (i.e., the prediction would equal the loss-augmented prediction).
