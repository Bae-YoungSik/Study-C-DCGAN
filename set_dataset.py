# import torchvision
# from torchvision import transforms


# transform = transforms.ToTensor()

# train_dataset = torchvision.datasets.FashionMNIST(
#     root="./data",
#     train=True,
#     download=True,
#     transform=transform
# )

# test_dataset = torchvision.datasets.FashionMNIST(
#     root="./data",
#     train=False,
#     download=True,
#     transform=transform
# )

# print("훈련 데이터 개수:", len(train_dataset))
# print("테스트 데이터 개수:", len(test_dataset))

import torch
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt


# 이미지 → Tensor
transform = transforms.ToTensor()


# 훈련 데이터
train_dataset = torchvision.datasets.FashionMNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)


# 테스트 데이터
test_dataset = torchvision.datasets.FashionMNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)


# 데이터 개수
print("훈련 데이터 개수:", len(train_dataset))
print("테스트 데이터 개수:", len(test_dataset))


# 첫 번째 데이터
image, label = train_dataset[0]


# 데이터 정보 출력
print("\n첫 번째 이미지 정보")
print("Tensor 크기:", image.shape)
print("Pixel 최소값:", image.min().item())
print("Pixel 최대값:", image.max().item())
print("Label:", label)


# 이미지 출력
plt.imshow(image.squeeze(), cmap="gray")
plt.title(f"Label: {label}")
plt.axis("off")
plt.show()

########
from torch.utils.data import DataLoader


batch_size = 64

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True
)


# 첫 번째 batch
images, labels = next(iter(train_loader))


print("\nDataLoader 테스트")
print("Image batch 크기:", images.shape)
print("Label batch 크기:", labels.shape)