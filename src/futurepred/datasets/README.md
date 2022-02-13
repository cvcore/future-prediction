# Datasets Definition for Training Perception / Prediction Network

## Supported Datasets for Semantic Segmentation

| Dataset                                | Training Size [1] | Validation Size | Test Size | Shape (H x W)   |
| -------------------------------------- | ----------------- | --------------- | --------- | --------------- |
| Cityscapes                             | 2975              | 500             | 1525      | 1024 x 2048     |
| Cityscapes Sequence (Efficient PS) [2] | 89250             | 15000           | 45750     | 1024 x 2048     |
| Cityscapes Sequence (Efficient PS) [2] | 89250             | 15000           | 45750     | 1024 x 2048 [3] |
| Mapillary                              | 18000             | 2000            | 0         | Various         |

**Note**

1. Number of images with ground truth labels.
2. These are extra Cityscapes sequence data with pseudo ground truth labels generated from a trained EfficientPS network. This dataset are used only for predicting the future semantic segmentation.
3. The pseudo ground truth labels for semantic segmentation are in resolution 256 x 512.
