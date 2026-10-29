from dataset import FacadesDataset
import matplotlib.pyplot as plt

dataset = FacadesDataset("dataset/facades/train")

input_image, target_image = dataset[0]

# Convert normalized tensors back to [0, 1]
input_image = (input_image * 0.5 + 0.5).permute(1, 2, 0)
target_image = (target_image * 0.5 + 0.5).permute(1, 2, 0)

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(input_image)
plt.title("Input Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(target_image)
plt.title("Target Image")
plt.axis("off")

plt.show()