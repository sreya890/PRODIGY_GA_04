import os
from PIL import Image

from torch.utils.data import Dataset
from torchvision import transforms


class FacadesDataset(Dataset):

    def __init__(self, root_dir):
        self.root_dir = root_dir

        self.image_files = sorted([
            file for file in os.listdir(root_dir)
            if file.lower().endswith(".jpg")
        ])

        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.5, 0.5, 0.5),
                (0.5, 0.5, 0.5)
            )
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):

        image_name = self.image_files[index]
        image_path = os.path.join(self.root_dir, image_name)

        image = Image.open(image_path).convert("RGB")

        width, height = image.size
        middle = width // 2

        # RIGHT HALF: Architectural segmentation map
        input_image = image.crop(
            (middle, 0, width, height)
        )

        # LEFT HALF: Real building photograph
        target_image = image.crop(
            (0, 0, middle, height)
        )

        input_image = self.transform(input_image)
        target_image = self.transform(target_image)

        return input_image, target_image