# Image-to-Image Translation Using Pix2Pix GAN

## Project Overview

This project implements **Image-to-Image Translation using Pix2Pix**, a Conditional Generative Adversarial Network (cGAN).

The model is trained to convert architectural segmentation maps into realistic building images using the **Facades Dataset**.

## Objective

To develop a Pix2Pix GAN model that learns the mapping between paired images and generates realistic building photographs from architectural segmentation maps.

## Dataset

**Dataset:** Facades Dataset

The dataset contains paired images:
- **Input:** Architectural segmentation maps
- **Target:** Real building photographs

Images are resized to **256 × 256 pixels** and normalized before training.

## Model Architecture

Pix2Pix consists of two neural networks:

**1. Generator (U-Net)**
- Generates realistic building images from segmentation maps.
- Uses skip connections to preserve image details.

**2. Discriminator (PatchGAN)**
- Distinguishes real image pairs from generated image pairs.
- Helps the generator produce more realistic images.

## Technologies Used

- Python
- PyTorch
- Torchvision
- NumPy
- Pillow
- Matplotlib
- Google Colab (GPU Training)
- VS Code

## Project Structure

```text
pix2pix/
│
├── dataset.py
├── inspectdata.py
├── test_dataset.py
├── generator.py
├── discriminator.py
├── train.py
├── predict.py
├── requirements.txt
│
├── checkpoints_seg_to_photo/
│   └── generator.pth
│
└── predictions_seg_to_photo/
    ├── generated_building.png
    └── comparison.png
```

## Installation

**1. Clone the repository**

```bash
git clone <repository-url>
cd pix2pix
```

**2. Install dependencies**

```bash
pip install -r requirements.txt
```

## Usage

**1. Train the model**

```bash
python train.py
```

Training was performed using Google Colab with GPU acceleration.

**2. Generate images**

After placing the trained generator weights in `checkpoints_seg_to_photo/`, run:

```bash
python predict.py
```

The generated images will be saved in the `predictions_seg_to_photo/` folder.

## Results

The trained Pix2Pix model generates building images from architectural segmentation maps.

The results are visualized by comparing:
- Input segmentation map
- Generated building image
- Original building photograph

## Conclusion

This project demonstrates image-to-image translation using a Conditional GAN (Pix2Pix). The U-Net Generator and PatchGAN Discriminator work together to learn the relationship between architectural segmentation maps and real building photographs.
