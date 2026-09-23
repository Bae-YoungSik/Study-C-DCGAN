import torchvision
from torchvision import transforms


transform = transforms.ToTensor()

train_dataset = torchvision.datasets.FashionMNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

test_dataset = torchvision.datasets.FashionMNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

print("훈련 데이터 개수:", len(train_dataset))
print("테스트 데이터 개수:", len(test_dataset))