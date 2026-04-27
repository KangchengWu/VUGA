# VUGA: Viewport-Unaware Blind Omnidirectional Image Quality Assessment

This repository provides the official PyTorch implementation of the following paper:

**Yan J, Wu K, Hou J, et al. Viewport-Unaware Blind Omnidirectional Image Quality Assessment: A Unified and Generalized Approach[J]. IEEE Transactions on Multimedia, 2026.**

## Introduction

Blind Omnidirectional Image Quality Assessment (BOIQA) aims to predict the perceptual quality of omnidirectional images without using reference images. Existing viewport-aware BOIQA methods usually rely on viewport extraction or handcrafted sampling strategies, which may introduce additional computational cost and depend on the accuracy of viewport generation.

To address these limitations, this repository implements a viewport-unaware BOIQA model, named **VUGA**, which directly takes omnidirectional images as input and predicts their perceptual quality scores. The proposed method provides a unified and generalized framework for blind omnidirectional image quality assessment.

## File Structure

```bash
.
├── CMP.py          # CMP module used in the proposed framework
├── GA.py           # GA module used in the proposed framework
├── MyDataset.py    # Dataset loading and preprocessing
├── SDA.py          # SDA module used in the proposed framework
├── VUGA.py         # Main VUGA model architecture
├── config.py       # Configuration file for dataset paths and training settings
├── train.py        # Training script
└── utils.py        # Utility functions
```

## Requirements

The code is implemented with PyTorch. A recommended environment is:

```bash
conda create -n vuga python=3.9
conda activate vuga

pip install torch torchvision torchaudio
pip install numpy scipy pandas pillow opencv-python tqdm timm scikit-learn
```

Please make sure that the installed PyTorch version is compatible with your CUDA version.

## DCNv3 Installation

This project may require **DCNv3**. The DCNv3 operator can be downloaded from the official InternImage repository:

```bash
git clone https://github.com/OpenGVLab/InternImage.git
```

The DCNv3 source code is located at:

```bash
InternImage/classification/ops_dcnv3
```

The `dcnv3.py` file is located at:

```bash
InternImage/classification/ops_dcnv3/modules/dcnv3.py
```

You can copy the whole DCNv3 operator folder into this project:

```bash
cp -r InternImage/classification/ops_dcnv3 ./ops_dcnv3
```

Then compile DCNv3:

```bash
cd ops_dcnv3
sh make.sh
python test.py
```

If the compilation is successful, the test script should run correctly.

Alternatively, you can download the precompiled DCNv3 wheel files from:

```bash
https://github.com/OpenGVLab/InternImage/releases/tag/whl_files
```

Please select the wheel file that matches your Python version, PyTorch version, CUDA version, and operating system.


## Configuration

Before training, please modify the dataset path, training settings, and model parameters in:

```bash
config.py
```

Typical settings include:

```python
dataset_path = "path/to/dataset"
train_list = "path/to/train.txt"
test_list = "path/to/test.txt"

batch_size = 8
learning_rate = 1e-4
epochs = 50
```

Please adjust these parameters according to your experimental environment.

## Training

To train the model, run:

```bash
python train.py
```

The training script will load the dataset, build the VUGA model, and optimize it for blind omnidirectional image quality prediction.

## Evaluation

The commonly used evaluation metrics for image quality assessment include:

- **SRCC**: Spearman Rank-order Correlation Coefficient
- **PLCC**: Pearson Linear Correlation Coefficient
- **KRCC**: Kendall Rank-order Correlation Coefficient
- **RMSE**: Root Mean Squared Error

Please refer to `utils.py` or `train.py` for the implementation of evaluation metrics.

## Citation

If you find this repository useful for your research, please cite our paper:

```bibtex
@article{yan2026vuga,
  title={Viewport-Unaware Blind Omnidirectional Image Quality Assessment: A Unified and Generalized Approach},
  author={Yan, J. and Wu, K. and Hou, J. and others},
  journal={IEEE Transactions on Multimedia},
  year={2026}
}
```

## Acknowledgement

This project uses the DCNv3 operator from the InternImage project. We sincerely thank the authors of InternImage for their excellent open-source implementation.

```bash
https://github.com/OpenGVLab/InternImage
```

## License

This repository is released for academic research purposes only. Please contact the authors for other usage.
