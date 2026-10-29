import os
import torch
import matplotlib.pyplot as plt

from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

from generator import Generator


# --------------------------------
# Configuration
# --------------------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_PATH = "checkpoints_seg_to_photo/generator.pth"

TEST_IMAGE_PATH = "dataset/facades/test/1.jpg"

OUTPUT_DIR = "predictions_seg_to_photo"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# --------------------------------
# Load Generator
# --------------------------------

generator = Generator().to(DEVICE)

generator.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True
    )
)

generator.eval()

print("Trained generator loaded successfully!")


# --------------------------------
# Transform
# --------------------------------

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.5, 0.5, 0.5),
        (0.5, 0.5, 0.5)
    )
])


# --------------------------------
# Load Paired Test Image
# --------------------------------

image = Image.open(TEST_IMAGE_PATH).convert("RGB")

width, height = image.size
middle = width // 2

# RIGHT: Architectural segmentation map
input_image = image.crop(
    (middle, 0, width, height)
)

# LEFT: Real building photograph
target_image = image.crop(
    (0, 0, middle, height)
)


# --------------------------------
# Preprocess
# --------------------------------

input_tensor = transform(input_image).unsqueeze(0).to(DEVICE)

target_tensor = transform(target_image).unsqueeze(0)


# --------------------------------
# Generate Building
# --------------------------------

with torch.no_grad():
    generated_tensor = generator(input_tensor)


# --------------------------------
# Save Generated Image
# --------------------------------

save_image(
    generated_tensor,
    os.path.join(OUTPUT_DIR, "generated_building.png"),
    normalize=True,
    value_range=(-1, 1)
)


# --------------------------------
# Visualization
# --------------------------------

def tensor_to_image(tensor):

    tensor = tensor.detach().cpu().squeeze(0)

    tensor = tensor * 0.5 + 0.5

    tensor = tensor.clamp(0, 1)

    return tensor.permute(1, 2, 0).numpy()


input_display = tensor_to_image(input_tensor)

generated_display = tensor_to_image(generated_tensor)

target_display = tensor_to_image(target_tensor)


plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(input_display)
plt.title("Architectural Segmentation Map")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(generated_display)
plt.title("Generated Building")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(target_display)
plt.title("Real Building")
plt.axis("off")

plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "comparison.png"),
    dpi=150,
    bbox_inches="tight"
)

plt.show()

print("Prediction completed successfully!")