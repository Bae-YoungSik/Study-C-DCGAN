import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader


# 이미지 전처리
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


# 훈련 데이터셋
train_dataset = torchvision.datasets.FashionMNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)


# 테스트 데이터셋
test_dataset = torchvision.datasets.FashionMNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)


# DataLoader
batch_size = 64

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False
)


# image, label = train_dataset[0]

# print("이미지 크기:", image.shape)
# print("최소값:", image.min().item())
# print("최대값:", image.max().item())
# print("Label:", label)
