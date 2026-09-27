import os
import torch
import torchvision
from torchvision.utils import save_image

from model import Generator
from tester.classifier import Classifier


# =========================
# 기본 설정
# =========================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

noise_dim = 100
num_samples = 100

checkpoint_dir = "./checkpoint/checkpoint_Periodic"
output_dir = "./tester/C_test"
classifier_path = "./tester/classifier/classifier.pth"


os.makedirs(output_dir, exist_ok=True)


# =========================
# 평가할 Epoch 자동 검색
# =========================

epochs = sorted(
    int(
        filename.split("_")[1].split(".")[0]
    )
    for filename in os.listdir(checkpoint_dir)
    if filename.startswith("checkpoint_")
    and filename.endswith(".pth")
)


# =========================
# 모델 생성
# =========================

generator = Generator(
    noise_dim=noise_dim,
    num_classes=10,
    embedding_dim=10
).to(device)


classifier = Classifier().to(device)

classifier.load_state_dict(
    torch.load(
        classifier_path,
        map_location=device
    )
)

classifier.eval()


# =========================
# 그룹 정의
# =========================

clothing_group = {
    0, 2, 3, 4, 6
}

shoe_group = {
    5, 7, 9
}


def same_group(condition, prediction):

    if condition in clothing_group:
        return prediction in clothing_group

    if condition in shoe_group:
        return prediction in shoe_group

    return condition == prediction


# =========================
# Epoch별 평가
# =========================

for epoch in epochs:

    print()
    print("=" * 60)
    print(f"Epoch [{epoch}]")
    print("=" * 60)


    # =========================
    # Checkpoint 불러오기
    # =========================

    checkpoint_path = os.path.join(
        checkpoint_dir,
        f"checkpoint_{epoch}.pth"
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device
    )

    generator.load_state_dict(
        checkpoint["generator_state_dict"]
    )

    generator.eval()


    # =========================
    # 평가용 변수
    # =========================

    total_correct = 0
    total_same_group_error = 0
    total_different_group_error = 0

    total_confidence = 0.0
    total_count = 0


    # 조건별 평가
    condition_correct = [0] * 10
    condition_count = [0] * 10

    condition_predictions = [
        [0] * 10
        for _ in range(10)
    ]


    # =========================
    # 여러 번 생성 및 평가
    # =========================

    labels = torch.arange(
        10,
        device=device
    )


    for sample_index in range(num_samples):

        # 새로운 Noise 생성
        noise = torch.randn(
            10,
            noise_dim,
            device=device
        )


        # =========================
        # 이미지 생성
        # =========================

        with torch.no_grad():

            generated_images = generator(
                noise,
                labels
            )


        # =========================
        # 첫 번째 반복만 이미지 저장
        # =========================

        if sample_index == 0:

            save_path = os.path.join(
                output_dir,
                f"epoch_{epoch}.png"
            )

            save_image(
                generated_images,
                save_path,
                nrow=10,
                normalize=True,
                value_range=(-1, 1)
            )


        # =========================
        # Classifier 평가
        # =========================

        with torch.no_grad():

            outputs = classifier(
                generated_images
            )

            probabilities = torch.softmax(
                outputs,
                dim=1
            )

            predictions = torch.argmax(
                probabilities,
                dim=1
            )

            confidences = torch.max(
                probabilities,
                dim=1
            ).values


        # =========================
        # 결과 분석
        # =========================

        for condition in range(10):

            prediction = predictions[
                condition
            ].item()

            confidence = confidences[
                condition
            ].item()


            condition_count[
                condition
            ] += 1

            condition_predictions[
                condition
            ][prediction] += 1


            total_confidence += confidence
            total_count += 1


            # 정확한 분류
            if prediction == condition:

                total_correct += 1

                condition_correct[
                    condition
                ] += 1

            # 오분류
            else:

                if same_group(
                    condition,
                    prediction
                ):

                    total_same_group_error += 1

                else:

                    total_different_group_error += 1


    # =========================
    # 전체 평가 결과
    # =========================

    accuracy = (
        total_correct
        / total_count
        * 100
    )

    same_group_error = (
        total_same_group_error
        / total_count
        * 100
    )

    different_group_error = (
        total_different_group_error
        / total_count
        * 100
    )

    mean_confidence = (
        total_confidence
        / total_count
    )


    print()
    print(
        f"총 생성 이미지: {total_count}"
    )

    print(
        f"정확도: {accuracy:.2f}%"
    )

    print(
        f"같은 그룹 오분류: "
        f"{same_group_error:.2f}%"
    )

    print(
        f"다른 그룹 오분류: "
        f"{different_group_error:.2f}%"
    )

    print(
        f"평균 Confidence: "
        f"{mean_confidence:.4f}"
    )


    # =========================
    # 조건별 정확도
    # =========================

    print()
    print("조건별 정확도")

    for condition in range(10):

        condition_accuracy = (
            condition_correct[condition]
            / condition_count[condition]
            * 100
        )

        print(
            f"Condition {condition}: "
            f"{condition_accuracy:.2f}%"
        )


    # =========================
    # 조건별 예측 분포
    # =========================

    print()
    print("조건별 예측 분포")

    for condition in range(10):

        print(
            f"Condition {condition}: "
            f"{condition_predictions[condition]}"
        )