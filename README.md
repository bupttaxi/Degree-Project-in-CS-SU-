# CIFAR-10 Classification Task Based on ResNet-18 and D-RISE Salience Map Analysis

This repository is for the codes from my bachelor degree project in CS (SU) 2024 spring semester.

## Abstract

Deep learning, especially Convolutional Neural Networks (CNNs), has made significant achievements in image recognition and classification tasks. Residual Network (ResNet), a revolutionary neural network architecture developed from CNNs, excels in visual tasks such as image recognition and object detection. Given the complexity of deep learning networks and their operation as "blackboxes", the explainability of models has become a key issue in the field of AI. By using saliency maps to reveal key features during the model's processing, we can enhance our understanding and trust in the model's classification decisions.

This thesis employs the ResNet-18 architecture for image classification on the CIFAR-10 dataset, created by the Canadian Institute for Advanced Research. The study aims to enhance the model’s explainability by demonstrating the importance of variables through saliency maps. This thesis not only validates the generalization ability of the model but also successfully reproduces and optimizes the Detector Randomized Input Sampling for Explanation (D-RISE) algorithm applied to the ResNet-18 classification task. The D-RISE algorithm visually displays the key features captured by the model and reveals the model’s decision logic. Lastly, in the discussion section of the thesis, we propose different methods for the similarity score calculation in the D-RISE algorithm to enhance the mathematical explainability behind the algorithm.

## Environment

- Python 3.10.13
- CUDA 11.2
- cuDNN 8.1

## Required Libraries and Modules

To run this code, you may need the following libraries or modules:

- tensorflow
- numpy
- matplotlib.pyplot
- seaborn
- scipy.ndimage
- random

GPU computing is recommended to reduce the training time. When you run this code, you only need to run it in your IDE (e.g., Spyder), and the plots will be shown in the plot window in Spyder.

## Defining Similarity Score

You can define different methods to calculate the similarity score, e.g., using cross-entropy as similarity score

```python
def similarity_calculation(proposals, selected_label):
    """
    DRISE 3: generating the similarity score using cross entropy
    """
    cross_entropy_loss = -np.sum(selected_label * np.log(proposals + 1e-9), axis=1)  # 加上一个小的epsilon防止对0取对数
    return cross_entropy_loss.tolist()

def plot_3d_matrix(w_norm, p1):
    """
    and you need to change the upper bound and lower bound:
    """
    ...

    z_high = 1 * 5000 * p1  # upper bound
    z_low = 0.1 * 5000 * p1   # lower bound
    ...

```

## Citation

If you use this code in your research, please cite:

@misc{xt2024thesis,
  author = {Xin Tang},
  title = {CIFAR-10 Classification Task Based on ResNet-18 and D-RISE Salience Map Analysis},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/yourusername/your-repo-name}}
}

The D-RISE algorithm:

@inproceedings{petsiuk2021black,
  title={Black-box explanation of object detectors via saliency maps},
  author={Petsiuk, Vitali and Jain, Rajiv and Manjunatha, Varun and Morariu, Vlad I and Mehra, Ashutosh and Ordonez, Vicente and Saenko, Kate},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={11443--11452},
  year={2021}
}

## Contact
For any questions or issues, please contact bupttaxi@gmail.com.
