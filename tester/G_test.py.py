import os

import torch

from model import Generator


# =========================
# 기본 설정
# =========================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

noise_dim = 100
num_classes = 10
num_samples = 100


# =========================
# Checkpoint 경로
# =========================

checkpoint_dir = "./checkpoint/checkpoint_Periodic"


# =========================
# Checkpoint 목록
# =========================

checkpoint_files = [
    file
    for file in os.listdir(checkpoint_dir)
    if file.startswith("checkpoint_")
    and file.endswith(".pth")
]


checkpoint_files.sort(
    key=lambda file: int(
        file.split("_")[1].split(".")[0]
    )
)


if len(checkpoint_files) == 0:

    raise FileNotFoundError(
        "checkpoint_Periodic에 Checkpoint가 없습니다."
    )


print("확인할 Checkpoint:")

for checkpoint_file in checkpoint_files:

    print(checkpoint_file)


print()


# =========================
# Generator 생성
# =========================

generator = Generator().to(device)


# =========================
# Checkpoint별 측정
# =========================

results = []


for checkpoint_file in checkpoint_files:

    checkpoint_path = os.path.join(
        checkpoint_dir,
        checkpoint_file
    )

    print("=" * 70)

    print(
        "Checkpoint:",
        checkpoint_file
    )

    # -------------------------
    # Checkpoint 불러오기
    # -------------------------

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device
    )

    generator.load_state_dict(
        checkpoint["generator_state_dict"]
    )

    generator.eval()

    # -------------------------
    # Label별 다양성 측정
    # -------------------------

    label_results = []

    with torch.no_grad():

        for label in range(num_classes):

            # 같은 Label에 대해
            # 서로 다른 Noise 100개 생성
            noise = torch.randn(
                num_samples,
                noise_dim,
                device=device
            )

            labels = torch.full(
                (num_samples,),
                label,
                dtype=torch.long,
                device=device
            )

            # 이미지 생성
            images = generator(
                noise,
                labels
            )

            # -------------------------
            # 이미지 평균
            # -------------------------

            mean_image = images.mean(
                dim=0,
                keepdim=True
            )

            # -------------------------
            # Noise Diversity
            # -------------------------
            #
            # 각 이미지가 평균 이미지에서
            # 얼마나 떨어져 있는지 측정
            #

            diversity = (
                (images - mean_image) ** 2
            ).mean().item()

            label_results.append(
                diversity
            )

    # -------------------------
    # 전체 Label 평균
    # -------------------------

    average_diversity = sum(
        label_results
    ) / len(label_results)

    results.append(
        (
            checkpoint["epoch"],
            label_results,
            average_diversity
        )
    )

    # -------------------------
    # 결과 출력
    # -------------------------

    for label, diversity in enumerate(
        label_results
    ):

        print(
            f"Label {label}: "
            f"{diversity:.8f}"
        )

    print(
        f"Average: "
        f"{average_diversity:.8f}"
    )


# =========================
# 전체 결과
# =========================

print()
print("=" * 70)
print("Noise Diversity Summary")
print("=" * 70)

print(
    f"{'Epoch':>10} "
    f"{'Average Diversity':>20}"
)

print("-" * 70)


for epoch, label_results, average_diversity in results:

    print(
        f"{epoch:>10} "
        f"{average_diversity:>20.8f}"
    )


# =========================
# Label별 결과
# =========================

print()
print("=" * 70)
print("Label Diversity Summary")
print("=" * 70)

print(
    f"{'Epoch':>10} "
    f"{'0':>12} "
    f"{'1':>12} "
    f"{'2':>12} "
    f"{'3':>12} "
    f"{'4':>12} "
    f"{'5':>12} "
    f"{'6':>12} "
    f"{'7':>12} "
    f"{'8':>12} "
    f"{'9':>12}"
)

print("-" * 150)


for epoch, label_results, average_diversity in results:

    print(
        f"{epoch:>10} "
        + " ".join(
            f"{value:>12.8f}"
            for value in label_results
        )
    )