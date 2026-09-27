'''
Train 데이터에서 훈련/Validation 분리

Validation Loss를 매 epoch 측정

두 조건을 모두 만족하면 학습 종료
    전체 Validation Accuracy ≥ 97%
    10개 클래스 모두 Accuracy ≥ 95%

종료 후 독립적인 Test 데이터로 최종 성능 확인

최종 C를 classifier.pth로 저장

이후 G/D 평가에서 사용할 수 있도록 모델 구조는 단순한 CNN으로 유지
'''

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

# Random seed
seed = 42
torch.manual_seed(seed)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Image transform
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


# Fashion-MNIST dataset
full_train_dataset = datasets.FashionMNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.FashionMNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)


# Train / Validation split
train_size = int(len(full_train_dataset) * 0.9)
validation_size = len(full_train_dataset) - train_size

train_dataset, validation_dataset = random_split(
    full_train_dataset,
    [train_size, validation_size]
)


# DataLoader
batch_size = 64

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=batch_size,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False
)


# Classifier
class Classifier(nn.Module):

    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(

            # 28x28 -> 14x14
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # 14x14 -> 7x7
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # 64 x 7 x 7 -> 128
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),

            # 10 classes
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.model(x)


classifier = Classifier().to(device)


# Loss / Optimizer
criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    classifier.parameters(),
    lr=0.001
)


# Training settings
max_epochs = 50

target_accuracy = 0.97
target_class_accuracy = 0.95


# Training
for epoch in range(max_epochs):

    classifier.train()

    train_loss = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        output = classifier(images)

        loss = criterion(output, labels)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()


    # Validation
    classifier.eval()

    validation_loss = 0

    class_correct = [0] * 10
    class_total = [0] * 10

    confusion_matrix = torch.zeros(10, 10, dtype=torch.int64)

    with torch.no_grad():

        for images, labels in validation_loader:

            images = images.to(device)
            labels = labels.to(device)

            output = classifier(images)

            loss = criterion(output, labels)

            validation_loss += loss.item()

            predicted = output.argmax(dim=1)

            for label, prediction in zip(labels, predicted):

                class_total[label.item()] += 1

                if label == prediction:
                    class_correct[label.item()] += 1

                confusion_matrix[label.item(), prediction.item()] += 1
    print()
    print("Validation Confusion Matrix")
    print(confusion_matrix)


    validation_loss /= len(validation_loader)

    class_accuracy = [
        class_correct[i] / class_total[i]
        for i in range(10)
    ]

    validation_accuracy = sum(class_correct) / sum(class_total)

    min_class_accuracy = min(class_accuracy)


    print(
        f"Epoch [{epoch + 1}/{max_epochs}] "
        f"Train Loss: {train_loss / len(train_loader):.4f} "
        f"Validation Loss: {validation_loss:.4f} "
        f"Validation Accuracy: {validation_accuracy * 100:.2f}% "
        f"Min Class Accuracy: {min_class_accuracy * 100:.2f}%"
    )
    print(
        "Class Accuracy: "
        + ", ".join(
            f"\n{i}: { class_accuracy[i] * 100:.2f}%"
            for i in range(10)
        )
    )


    # Check training criteria
    if (
        validation_accuracy >= target_accuracy
        and min_class_accuracy >= target_class_accuracy
    ):

        print("Classifier training criteria satisfied.")

        break


# Final test evaluation
classifier.eval()

test_loss = 0

class_correct = [0] * 10
class_total = [0] * 10

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        output = classifier(images)

        loss = criterion(output, labels)

        test_loss += loss.item()

        predicted = output.argmax(dim=1)

        for label, prediction in zip(labels, predicted):

            class_total[label.item()] += 1

            if label == prediction:
                class_correct[label.item()] += 1


test_loss /= len(test_loader)

test_accuracy = sum(class_correct) / sum(class_total)

test_class_accuracy = [
    class_correct[i] / class_total[i]
    for i in range(10)
]


# Final results
print()
print("Final Test Results")
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

print("Class Accuracy:")

for i in range(10):

    print(
        f"Class {i}: "
        f"{test_class_accuracy[i] * 100:.2f}%"
    )


# Save classifier
torch.save(
    classifier.state_dict(),
    "./classifier/classifier.pth"
)

print()
print("Classifier saved.")