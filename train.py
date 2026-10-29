import os

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

from dataset import FacadesDataset
from generator import Generator
from discriminator import Discriminator


# -----------------------------
# Configuration
# -----------------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LEARNING_RATE = 0.0002
BATCH_SIZE = 1
EPOCHS = 20
LAMBDA_L1 = 100

TRAIN_DIR = "dataset/facades/train"

os.makedirs("checkpoints", exist_ok=True)
os.makedirs("outputs", exist_ok=True)


# -----------------------------
# Load Dataset
# -----------------------------

train_dataset = FacadesDataset(TRAIN_DIR)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

print("Training images:", len(train_dataset))
print("Device:", DEVICE)


# -----------------------------
# Initialize Models
# -----------------------------

generator = Generator().to(DEVICE)
discriminator = Discriminator().to(DEVICE)


# -----------------------------
# Loss Functions
# -----------------------------

adversarial_loss = nn.BCEWithLogitsLoss()
l1_loss = nn.L1Loss()


# -----------------------------
# Optimizers
# -----------------------------

optimizer_G = optim.Adam(
    generator.parameters(),
    lr=LEARNING_RATE,
    betas=(0.5, 0.999)
)

optimizer_D = optim.Adam(
    discriminator.parameters(),
    lr=LEARNING_RATE,
    betas=(0.5, 0.999)
)


# -----------------------------
# Training Loop
# -----------------------------

for epoch in range(EPOCHS):

    generator.train()
    discriminator.train()

    total_g_loss = 0
    total_d_loss = 0

    progress_bar = tqdm(
        train_loader,
        desc=f"Epoch [{epoch + 1}/{EPOCHS}]"
    )

    for input_image, target_image in progress_bar:

        input_image = input_image.to(DEVICE)
        target_image = target_image.to(DEVICE)

        # -----------------------------
        # Train Discriminator
        # -----------------------------

        fake_image = generator(input_image)

        real_output = discriminator(
            input_image,
            target_image
        )

        fake_output = discriminator(
            input_image,
            fake_image.detach()
        )

        real_labels = torch.ones_like(real_output)
        fake_labels = torch.zeros_like(fake_output)

        loss_D_real = adversarial_loss(
            real_output,
            real_labels
        )

        loss_D_fake = adversarial_loss(
            fake_output,
            fake_labels
        )

        loss_D = 0.5 * (
            loss_D_real + loss_D_fake
        )

        optimizer_D.zero_grad()
        loss_D.backward()
        optimizer_D.step()

        # -----------------------------
        # Train Generator
        # -----------------------------

        fake_output = discriminator(
            input_image,
            fake_image
        )

        loss_G_GAN = adversarial_loss(
            fake_output,
            torch.ones_like(fake_output)
        )

        loss_G_L1 = l1_loss(
            fake_image,
            target_image
        )

        loss_G = (
            loss_G_GAN +
            LAMBDA_L1 * loss_G_L1
        )

        optimizer_G.zero_grad()
        loss_G.backward()
        optimizer_G.step()

        total_g_loss += loss_G.item()
        total_d_loss += loss_D.item()

        progress_bar.set_postfix({
            "G_Loss": f"{loss_G.item():.4f}",
            "D_Loss": f"{loss_D.item():.4f}"
        })

    # -----------------------------
    # Save Generated Images
    # -----------------------------

    generator.eval()

    with torch.no_grad():

        sample_input, sample_target = next(
            iter(train_loader)
        )

        sample_input = sample_input.to(DEVICE)

        generated = generator(sample_input)

        comparison = torch.cat(
            [
                sample_input.cpu(),
                generated.cpu(),
                sample_target
            ],
            dim=0
        )

        save_image(
            comparison,
            f"outputs/epoch_{epoch + 1}.png",
            normalize=True,
            value_range=(-1, 1)
        )

    # -----------------------------
    # Save Model Checkpoints
    # -----------------------------

    torch.save(
        generator.state_dict(),
        "checkpoints/generator.pth"
    )

    torch.save(
        discriminator.state_dict(),
        "checkpoints/discriminator.pth"
    )

    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"G Loss: {total_g_loss / len(train_loader):.4f} "
        f"D Loss: {total_d_loss / len(train_loader):.4f}"
    )

print("Training Completed!")