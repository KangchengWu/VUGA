# VUGA: Viewport-Unaware Blind Omnidirectional Image Quality Assessment

This repository provides the PyTorch implementation of the following paper:

**Yan J, Wu K, Hou J, et al. Viewport-Unaware Blind Omnidirectional Image Quality Assessment: A Unified and Generalized Approach[J]. IEEE Transactions on Multimedia, 2026.**

## Introduction

Blind Omnidirectional Image Quality Assessment (BOIQA) aims to automatically predict the perceptual quality of omnidirectional images without using reference images. Existing viewport-aware BOIQA methods usually rely on viewport extraction or handcrafted sampling strategies, which may introduce additional computational cost and depend on the quality of viewport generation.

This repository implements a viewport-unaware BOIQA framework, named VUGA, which directly takes omnidirectional images as input and predicts their perceptual quality scores. The model is designed to provide a unified and generalized solution for blind omnidirectional image quality assessment.

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
Requirements
The code is implemented with PyTorch. A recommended environment is:

conda create -n vuga python=3.9
conda activate vuga

pip install torch torchvision torchaudio
pip install numpy scipy pandas pillow opencv-python tqdm timm scikit-learn

DCNv3 Installation

This project may require DCNv3. The DCNv3 operator can be downloaded from the official InternImage repository:git clone https://github.com/OpenGVLab/InternImage.git
The DCNv3 source code is located at:InternImage/classification/ops_dcnv3
You can copy the DCNv3 operator folder into this project:cp -r InternImage/classification/ops_dcnv3 ./ops_dcnv3
Then compile DCNv3:cd ops_dcnv3
sh make.sh
python test.py
Citation

If you find this repository useful for your research, please cite our paper:
@article{yan2026vuga,
  title={Viewport-Unaware Blind Omnidirectional Image Quality Assessment: A Unified and Generalized Approach},
  author={Yan, J. and Wu, K. and Hou, J. and others},
  journal={IEEE Transactions on Multimedia},
  year={2026}
}
