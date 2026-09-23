import torch
import torch.nn as nn


class Generator(nn.Module):

    def __init__(
        self,
        noise_dim=100,
        num_classes=10,
        embedding_dim=10
    ):
        super().__init__()

        # Label → Embedding
        self.label_embedding = nn.Embedding(
            num_classes,
            embedding_dim
        )

        # Noise + Label → 128 × 7 × 7
        self.fc = nn.Linear(
            noise_dim + embedding_dim,
            128 * 7 * 7
        )

        # 7×7 → 14×14 → 28×28
        self.model = nn.Sequential(

            nn.BatchNorm2d(128),

            nn.ConvTranspose2d(
                128,
                64,
                kernel_size=4,
                stride=2,
                padding=1
            ),

            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(
                64,
                1,
                kernel_size=4,
                stride=2,
                padding=1
            ),

            nn.Tanh()
        )

    def forward(self, noise, labels):

        # Label embedding
        label = self.label_embedding(labels)

        # Noise와 Label 결합
        x = torch.cat(
            [noise, label],
            dim=1
        )

        # Fully Connected
        x = self.fc(x)

        # 128 × 7 × 7로 변환
        x = x.view(
            x.size(0),
            128,
            7,
            7
        )

        # 이미지 생성
        image = self.model(x)

        return image


class Discriminator(nn.Module):

    def __init__(
        self,
        num_classes=10,
        embedding_dim=10
    ):
        super().__init__()

        # Label → Embedding
        self.label_embedding = nn.Embedding(
            num_classes,
            embedding_dim
        )

        # Image → Feature
        self.image_model = nn.Sequential(

            nn.Conv2d(
                1,
                64,
                kernel_size=4,
                stride=2,
                padding=1
            ),

            nn.LeakyReLU(
                0.2,
                inplace=True
            ),

            nn.Conv2d(
                64,
                128,
                kernel_size=4,
                stride=2,
                padding=1
            ),

            nn.BatchNorm2d(128),

            nn.LeakyReLU(
                0.2,
                inplace=True
            )
        )

        # Image feature + Label → Real/Fake
        self.fc = nn.Linear(
            128 * 7 * 7 + embedding_dim,
            1
        )

    def forward(self, image, labels):

        # 이미지 특징 추출
        x = self.image_model(image)

        # Flatten
        x = x.view(
            x.size(0),
            -1
        )

        # Label embedding
        label = self.label_embedding(labels)

        # Image feature + Label 결합
        x = torch.cat(
            [x, label],
            dim=1
        )

        # Real / Fake 판별
        output = self.fc(x)

        return output