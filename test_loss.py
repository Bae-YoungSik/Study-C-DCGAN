import torch
import torch.nn as nn

from model import Generator, Discriminator


# Device 설정
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("사용 장치:", device)


# 기본 설정
batch_size = 64
noise_dim = 100


# 모델 생성
generator = Generator().to(device)
discriminator = Discriminator().to(device)


# Loss Function
criterion = nn.BCEWithLogitsLoss()


# 테스트용 데이터
noise = torch.randn(
    batch_size,
    noise_dim,
    device=device
)

labels = torch.randint(
    0,
    10,
    (batch_size,),
    device=device
)


# Generator
fake_images = generator(
    noise,
    labels
)


# Discriminator
fake_output = discriminator(
    fake_images,
    labels
)


# Target 생성
fake_targets = torch.zeros_like(fake_output)


# Loss 계산
fake_loss = criterion(
    fake_output,
    fake_targets
)


print("\nGenerator / Discriminator 테스트")
print("생성 이미지 크기:", fake_images.shape)
print("Discriminator 출력 크기:", fake_output.shape)

print("\nTarget 테스트")
print("Fake Target 크기:", fake_targets.shape)
print("Fake Target:", fake_targets[:5])

print("\nLoss 테스트")
print("Fake Loss:", fake_loss.item())