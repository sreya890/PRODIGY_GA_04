import torch
import torch.nn as nn


class Discriminator(nn.Module):

    def __init__(self, in_channels=3):
        super().__init__()

        def discriminator_block(
            input_channels,
            output_channels,
            stride=2,
            normalize=True
        ):

            layers = [
                nn.Conv2d(
                    input_channels,
                    output_channels,
                    kernel_size=4,
                    stride=stride,
                    padding=1,
                    bias=not normalize
                )
            ]

            if normalize:
                layers.append(
                    nn.BatchNorm2d(output_channels)
                )

            layers.append(
                nn.LeakyReLU(0.2, inplace=True)
            )

            return layers

        self.model = nn.Sequential(

            # Input + Target = 6 channels
            *discriminator_block(6, 64, normalize=False),

            *discriminator_block(64, 128),

            *discriminator_block(128, 256),

            *discriminator_block(
                256, 512, stride=1
            ),

            nn.Conv2d(
                512,
                1,
                kernel_size=4,
                stride=1,
                padding=1
            )
        )

    def forward(self, input_image, target_image):

        # Concatenate input and target images
        combined = torch.cat(
            [input_image, target_image],
            dim=1
        )

        return self.model(combined)


# Test discriminator
if __name__ == "__main__":

    discriminator = Discriminator()

    input_image = torch.randn(1, 3, 256, 256)

    target_image = torch.randn(1, 3, 256, 256)

    output = discriminator(input_image, target_image)

    print("Input shape:", input_image.shape)
    print("Target shape:", target_image.shape)
    print("Discriminator output shape:", output.shape)