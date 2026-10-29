from PIL import Image
import matplotlib.pyplot as plt
import os

# Path to one training image
image_path = "dataset/facades/train/100.jpg"

# Load image
image = Image.open(image_path)

print("Image size:", image.size)
print("Image mode:", image.mode)

# Display the original paired image
plt.figure(figsize=(12, 6))
plt.imshow(image)
plt.axis("off")
plt.title("Original Paired Image")
plt.show()