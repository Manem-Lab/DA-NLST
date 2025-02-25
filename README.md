<!-- #region -->
# Data augmented lung cancer prediction framework using the nested case control NLST cohort

[[Paper & Supplementary](https://www.frontiersin.org/journals/oncology/articles/10.3389/fonc.2025.1492758/full)] [[Dataset & Pretrained models](https://drive.google.com/drive/folders/1ioszTP-vkjJIcJxoANhQRSP6VMfRb6Oo?usp=sharing)]

## Abstract
**Purpose:** In the context of lung cancer screening, the scarcity of well-labeled medical images poses a significant challenge to implement supervised learning-based deep learning methods. While data augmentation is an effective technique for countering the difficulties caused by insufficient data, it has not been fully explored in the context of lung cancer screening. In this research study, we analyzed the state-of-the-art (SOTA) data augmentation techniques for lung cancer binary prediction.

**Methods:** To comprehensively evaluate the efficiency of data augmentation approaches, we considered the nested case control National Lung Screening Trial (NLST) cohort comprising of 253 individuals who had the commonly used CT scans without contrast. The CT scans were pre-processed into three-dimensional volumes based on the lung nodule annotations. Subsequently, we evaluated five basic (online) and two generative model-based offline data augmentation methods with ten state-of-the-art (SOTA) 3D deep learning-based lung cancer prediction models.

**Results:** Our results demonstrated that the performance improvement by data augmentation was highly dependent on approach used. The Cutmix method resulted in the highest average performance improvement across all three metrics: 1.07%, 3.29%, 1.19% for accuracy, F1 score and AUC, respectively. MobileNetV2 with a simple data augmentation approach achieved the best AUC of 0.8719 among all lung cancer predictors, demonstrating a 7.62% improvement compared to baseline. Furthermore, the MED-DDPM data augmentation approach was able to improve prediction performance by rebalancing the training set and adding moderately synthetic data.

**Conclusions:** The effectiveness of online and offline data augmentation methods were highly sensitive to the prediction model, highlighting the importance of carefully selecting the optimal data augmentation method. Our findings suggest that certain traditional methods can provide more stable and higher performance compared to SOTA online data augmentation approaches. Overall, these results offer meaningful insights for the development and clinical integration of data augmented deep learning tools for lung cancer screening.
## Installation
```
git clone https://github.com/Manem-Lab/DL-NLST
cd DL-NLST
conda create -n dl-nlst python=3.9
conda activate dl-nlst
conda install pytorch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install monai==1.3.0
pip install ./WAMA_Modules/
pip install matplotlib pandas scikit-learn tensorboard openpyxl nibabel scikit-image
pip install numpy==1.26.4
pip install tensorflow==2.16.1
```

## Preparation
Download the prepared dataset and pretrained models from [Google drive](https://drive.google.com/drive/folders/13EO2hUXm-rwUhlq_qgS-imChn6Ok5vvg?usp=sharing). Unzip the NLST.zip and the pretrained_weights.zip into ./dataset folder and the root path, the directory structure should look similar as follows:
```
DL-NLST/
│── dataset/
│   ├── NLST/            # Prepared dataset
│   │   ├── 3D/
│   ...
│── pretrained_weights/  # Pretrained models
│── train_da.py
│── README.md
...
```

## Training for 3D models with data augmention
```
python train_da.py --model ResNet18 --cross_val --da random
```

## Citation
If our work contributes to your research, please cite as follows:
```
@ARTICLE{10.3389/fonc.2025.1492758,
AUTHOR={Jiang, Yifan  and Manem, Venkata S. K. },
TITLE={Data augmented lung cancer prediction framework using the nested case control NLST cohort},
JOURNAL={Frontiers in Oncology},
VOLUME={15},
YEAR={2025},
URL={https://www.frontiersin.org/journals/oncology/articles/10.3389/fonc.2025.1492758},
DOI={10.3389/fonc.2025.1492758},
ISSN={2234-943X},
}
```

## Acknowledgements
This project is developed based on the following code repositories:
1. [WAMA_Modules](https://github.com/WAMAWAMA/WAMA_Modules)
2. [Efficient-3DCNNs](https://github.com/okankop/Efficient-3DCNNs)

We are very grateful for their contributions to the community.

