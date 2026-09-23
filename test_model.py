import torch

from model import Generator, Discriminator


# Device 설정
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("사용 장치:", device)


# 모델 생성
generator = Generator().to(device)
discriminator = Discriminator().to(device)


# 테스트용 데이터
batch_size = 64
noise_dim = 100

noise = torch.randn(
    batch_size,
    noise_dim
).to(device)

labels = torch.randint(
    0,
    10,
    (batch_size,)
).to(device)


# Generator 테스트
fake_images = generator(
    noise,
    labels
)


print("\nGenerator 테스트")
print("Noise 크기:", noise.shape)
print("Label 크기:", labels.shape)
print("생성 이미지 크기:", fake_images.shape)
print("생성 이미지 최소값:", fake_images.min().item())
print("생성 이미지 최대값:", fake_images.max().item())


# Discriminator 테스트
output = discriminator(
    fake_images,
    labels
)


print("\nDiscriminator 테스트")
print("입력 이미지 크기:", fake_images.shape)
print("Label 크기:", labels.shape)
print("출력 크기:", output.shape)
print("출력값:", output[:5])