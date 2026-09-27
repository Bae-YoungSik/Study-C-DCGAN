import os
import torch

from model import Discriminator


# =========================
# 환경 설정
# =========================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# =========================
# 확인할 Checkpoint
# =========================

checkpoint_groups = [

    # third
    (
        "./checkpoint/3. third/checkpoint_Periodic",
        1,
        500
    ),

    (
        "./checkpoint/3. third/checkpoint_Periodic",
        5600,
        float("inf")
    ),

    # fourth
    (
        "./checkpoint/4. fourth/checkpoint_Periodic",
        1,
        float("inf")
    )
]


# =========================
# Checkpoint 확인
# =========================

for checkpoint_root, min_epoch, max_epoch in checkpoint_groups:

    files = os.listdir(checkpoint_root)

    for file in sorted(files):

        if not file.endswith(".pth"):
            continue

        # Epoch 번호 추출
        try:
            epoch = int(
                file.replace(
                    "checkpoint_",
                    ""
                ).replace(
                    ".pth",
                    ""
                )
            )

        except ValueError:
            continue

        # Epoch 범위 확인
        if epoch < min_epoch or epoch > max_epoch:
            continue

        checkpoint_path = os.path.join(
            checkpoint_root,
            file
        )

        checkpoint = torch.load(
            checkpoint_path,
            map_location=device
        )

        discriminator = Discriminator().to(device)

        discriminator.load_state_dict(
            checkpoint["discriminator_state_dict"]
        )

        # FC 가중치
        fc_weight = discriminator.fc.weight

        # 이미지 특징에 해당하는 가중치
        image_weights = fc_weight[:, :-10]

        # Label embedding에 해당하는 가중치
        label_weights = fc_weight[:, -10:]

        # Norm 계산
        image_norm = image_weights.norm().item()
        label_norm = label_weights.norm().item()

        # Label 가중치와 Image 가중치의 상대적인 크기
        ratio = label_norm / image_norm

        print(
            f"{checkpoint_path}"
        )

        print(
            f"  Image FC weight norm : {image_norm:.6f}"
        )

        print(
            f"  Label FC weight norm : {label_norm:.6f}"
        )

        print(
            f"  Label / Image ratio  : {ratio:.6f}"
        )

        print()