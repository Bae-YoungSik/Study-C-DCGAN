''''
| 평가 대상   | 측정값                     |
|-----------|----------------------------|
| G         | 조건 일치율                 |
| G         | Classifier confidence 평균 |
| G         | Confidence 표준편차         |
| G         | 클래스별 조건 일치율         |
| D         | 올바른 label의 평균 점수     |
| D         | 잘못된 label의 평균 점수     |
| D         | 두 점수의 차이              |

주의할 점

    노이즈 벡터는 모든 과정에서 동일하게 사용할 것
    한 CheckPoint 마다 100개의 노이즈 벡터를 만들어 결과는 평균과 표준편차를 구할것
    평가에 이용할 실제 데이터를 고정할 것
    모델들은 평가 모드에 있을 것[generator.eval(), discriminator.eval(), classifier.eval()]
    D의 출력은 sigmoid를 취하지 않은 상태이므로 주의할 것
    클레스의 개수가 10개 이므로 하나의 이미지에 대해 10번의 검사를 모두 진행하는 것은 연산량이 많으니 적절히 조절할 것
    G가 이미지를 생성할 때 클레스간 생성하는 이미지의 개수는 동일할 것
    Classifier C의 오류 또한 평가에 영향을 주기에 참고만 할 것


구현

    3. third, 4. fourth의 checkpoint_Periodic 전체 평가
    G: 클래스당 100개 noise × 10개 클래스 = 1,000개 생성 이미지
    모든 체크포인트에서 동일한 noise와 label 사용
    C로 G의 조건 일치율 / confidence 평균 / 표준편차 / 클래스별 일치율 측정
    D: 고정된 Fashion-MNIST test subset 사용
    D에 정답 label과 오답 label을 넣어 조건 반응 측정
    D의 정답 label 평균 logit / 오답 label 평균 logit / 차이 기록
    .eval() + torch.no_grad() 사용
    결과를 evaluation_result.csv로 저장
    모드 붕괴/다양성 평가는 이번 코드에서 제외

    


    3차 / 4차 Checkpoint
        │
        ├──────────────┐
        ↓              ↓
      Generator      Discriminator
        │              │
        ↓              ↓
   이미지 1,000장    실제 이미지
        │              │
        ↓              ↓
   Classifier C     정답 label
        │              │
        ↓              ↓
  G 조건 일치율       D 조건 점수
        │              │
        └──────┬───────┘
               ↓
        evaluation_result.csv
'''


import os
import csv

import torch
import torch.nn as nn

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset



# CUDA 사용이 가능하면 GPU를 사용하고, 그렇지 않으면 CPU를 사용
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# 생성기
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



# 판별기
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



# 분류기
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



# 평가할 체크포인트 폴더
checkpoint_folders = {
    "fourth": "./checkpoint/4. fourth/checkpoint_Periodic",
    "fifth": "./checkpoint/checkpoint_Periodic"
}

# 클래스당 생성할 이미지 수
samples_per_class = 100

# 클래스당 판별기에 사용할 실제 이미지 수
real_samples_per_class = 100

# 평가에서 사용할 난수 고정
seed = 42
torch.manual_seed(seed)



# 각 클래스를 100번씩 생성하도록
# 0, 0, ..., 1, 1, ..., 9, 9 형태의 라벨을 생성
evaluation_labels = torch.arange(10).repeat_interleave(
    samples_per_class
).to(device)



# 모든 체크포인트에서 동일하게 사용할 무작위 잡음 생성
# 이렇게 하면 체크포인트 간 성능을 공정하게 비교할 수 있음
evaluation_noise = torch.randn(
    len(evaluation_labels),
    100,
    device=device
)



# Fashion-MNIST 테스트 데이터셋 불러오기
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

test_dataset = datasets.FashionMNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)



# 각 클래스에서 평가에 사용할 실제 이미지 선택
test_indices = []

for class_id in range(10):

    class_indices = [
        index
        for index, (_, label) in enumerate(test_dataset)
        if label == class_id
    ]

    test_indices.extend(
        class_indices[:real_samples_per_class]
    )

real_subset = Subset(
    test_dataset,
    test_indices
)



# 실제 이미지 평가를 위한 DataLoader
real_loader = DataLoader(
    real_subset,
    batch_size=64,
    shuffle=False
)



# 학습이 완료된 분류기 불러오기
classifier = Classifier().to(device)

classifier.load_state_dict(
    torch.load(
        "./tester/classifier/classifier.pth",
        map_location=device
    )
)

classifier.eval()



# 생성기의 조건부 생성 성능 평가
def evaluate_generator(generator):

    generator.eval()

    with torch.no_grad():

        # 고정된 잡음과 라벨을 사용하여 이미지 생성
        generated_images = generator(
            evaluation_noise,
            evaluation_labels
        )

        # 분류기를 이용하여 생성 이미지의 클래스를 예측
        outputs = classifier(generated_images)

        # 각 클래스에 대한 확률 계산
        probabilities = torch.softmax(outputs, dim=1)

        # 가장 높은 확률을 가진 클래스를 예측 결과로 사용
        predicted_labels = probabilities.argmax(dim=1)

        # 예측된 클래스의 확률을 신뢰도로 사용
        confidence = probabilities.max(dim=1).values

        # 생성 요청 라벨과 분류 결과가 같은지 확인
        matches = predicted_labels == evaluation_labels

        # 전체 조건 일치율 계산
        condition_match_rate = matches.float().mean().item()

        # 평균 신뢰도 계산
        confidence_mean = confidence.mean().item()

        # 신뢰도의 표준편차 계산
        confidence_std = confidence.std().item()

        # 클래스별 조건 일치율 계산
        class_match_rates = {}

        for class_id in range(10):

            class_mask = evaluation_labels == class_id

            class_match_rates[class_id] = (
                matches[class_mask]
                .float()
                .mean()
                .item()
            )

    return (
        condition_match_rate,
        confidence_mean,
        confidence_std,
        class_match_rates
    )



# 판별기의 조건부 판별 성능 평가
def evaluate_discriminator(discriminator):

    discriminator.eval()

    correct_scores = []
    wrong_scores = []

    with torch.no_grad():

        for images, labels in real_loader:

            images = images.to(device)
            labels = labels.to(device)

            # 실제 이미지와 올바른 클래스의 조합에 대한 점수
            correct_output = discriminator(
                images,
                labels
            )

            # 의도적으로 잘못된 클래스를 지정
            wrong_labels = (labels + 1) % 10

            # 실제 이미지와 잘못된 클래스의 조합에 대한 점수
            wrong_output = discriminator(
                images,
                wrong_labels
            )

            correct_scores.extend(
                correct_output.squeeze(1).cpu().tolist()
            )

            wrong_scores.extend(
                wrong_output.squeeze(1).cpu().tolist()
            )

    # 올바른 label의 점수가 더 높은 비율
    condition_accuracy = sum(
        correct > wrong
        for correct, wrong
        in zip(correct_scores, wrong_scores)
    ) / len(correct_scores)

    # 올바른 조건의 평균 점수
    correct_score = sum(correct_scores) / len(correct_scores)

    # 잘못된 조건의 평균 점수
    wrong_score = sum(wrong_scores) / len(wrong_scores)

    # 올바른 조건과 잘못된 조건의 점수 차이
    condition_score_difference = (
        correct_score - wrong_score
    )

    print(
        "D condition accuracy:",
        condition_accuracy
    )

    return (
        correct_score,
        wrong_score,
        condition_score_difference
    )



# 평가 결과를 저장할 리스트
results = []



# 실험별 체크포인트 평가
for experiment, folder in checkpoint_folders.items():

    # 체크포인트 파일 목록 가져오기
    checkpoint_files = [
        file
        for file in os.listdir(folder)
        if file.startswith("checkpoint_")
        and file.endswith(".pth")
    ]

    # 체크포인트 번호를 기준으로 정렬
    checkpoint_files.sort(
        key=lambda x: int(
            x.split("_")[1].split(".")[0]
        )
    )


    for checkpoint_file in checkpoint_files:

        # 체크포인트 전체 경로
        checkpoint_path = os.path.join(
            folder,
            checkpoint_file
        )

        # 체크포인트 불러오기
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device
        )

        # 생성기 생성 후 가중치 불러오기
        generator = Generator().to(device)

        generator.load_state_dict(
            checkpoint["generator_state_dict"]
        )

        # 판별기 생성 후 가중치 불러오기
        discriminator = Discriminator().to(device)

        discriminator.load_state_dict(
            checkpoint["discriminator_state_dict"]
        )


        # 생성기 평가
        (
            G_condition_match_rate,
            G_confidence_mean,
            G_confidence_std,
            G_class_match_rates
        ) = evaluate_generator(generator)


        # 판별기 평가
        (
            D_correct_score,
            D_wrong_score,
            D_condition_score_difference
        ) = evaluate_discriminator(discriminator)


        # 체크포인트 번호 추출
        epoch = int(
            checkpoint_file.split("_")[1].split(".")[0]
        )


        # 평가 결과 저장
        result = {
            "experiment": experiment,
            "epoch": epoch,

            "G_condition_match_rate":
                G_condition_match_rate,

            "G_confidence_mean":
                G_confidence_mean,

            "G_confidence_std":
                G_confidence_std,

            "D_correct_score":
                D_correct_score,

            "D_wrong_score":
                D_wrong_score,

            "D_condition_score_difference":
                D_condition_score_difference
        }


        # 클래스별 생성 조건 일치율 추가
        for class_id in range(10):

            result[
                f"G_class_{class_id}_match_rate"
            ] = G_class_match_rates[class_id]


        results.append(result)



# 평가 결과를 CSV 파일로 저장
output_file = "./evaluation_result.csv"

fieldnames = results[0].keys()

with open(
    output_file,
    "w",
    newline="",
    encoding="utf-8-sig"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames
    )

    writer.writeheader()
    writer.writerows(results)


print(f"평가 완료: {output_file}")