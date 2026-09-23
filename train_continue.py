import os
import torch
import torch.nn as nn

from dataset import train_loader, test_loader
from model import Generator, Discriminator


# 기본 설정
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

noise_dim = 100

# 이번 실행에서 추가로 학습할 epoch 수
additional_epochs = 900

# Checkpoint 설정
checkpoint_dir = "checkpoint"

periodic_dir = os.path.join(
    checkpoint_dir,
    "checkpoint_Periodic"
)

latest_dir = os.path.join(
    checkpoint_dir,
    "checkpoint_Latest"
)

# Checkpoint 폴더 확인
if not os.path.exists(latest_dir):

    raise FileNotFoundError(
        "checkpoint_Latest 폴더를 찾을 수 없습니다."
    )

# 가장 최근 Checkpoint 찾기
checkpoint_files = [
    file
    for file in os.listdir(latest_dir)
    if file.endswith(".pth")
]

if len(checkpoint_files) == 0:

    raise FileNotFoundError(
        "checkpoint_Latest에 checkpoint가 없습니다."
    )

checkpoint_files.sort(
    key=lambda file: int(
        file.split("_")[1].split(".")[0]
    )
)

latest_file = checkpoint_files[-1]

latest_path = os.path.join(
    latest_dir,
    latest_file
)

print("불러올 Checkpoint:")
print(latest_path)

# 모델 생성
generator = Generator().to(device)
discriminator = Discriminator().to(device)

# Loss Function
criterion = nn.BCEWithLogitsLoss()

# Optimizer
optimizer_G = torch.optim.Adam(
    generator.parameters(),
    lr=0.0002,
    betas=(0.5, 0.999)
)

optimizer_D = torch.optim.Adam(
    discriminator.parameters(),
    lr=0.0002,
    betas=(0.5, 0.999)
)

# Checkpoint 불러오기
checkpoint = torch.load(
    latest_path,
    map_location=device
)

generator.load_state_dict(
    checkpoint["generator_state_dict"]
)

discriminator.load_state_dict(
    checkpoint["discriminator_state_dict"]
)

optimizer_G.load_state_dict(
    checkpoint["optimizer_G_state_dict"]
)

optimizer_D.load_state_dict(
    checkpoint["optimizer_D_state_dict"]
)

start_epoch = checkpoint["epoch"]

end_epoch = start_epoch + additional_epochs

print()
print("Checkpoint 불러오기 완료")
print("현재 Epoch:", start_epoch)
print("학습 종료 Epoch:", end_epoch)

# 테스트 함수

def test_discriminator():

    generator.eval()
    discriminator.eval()

    total_loss = 0

    with torch.no_grad():

        for images, labels in test_loader:

            real_images = images.to(device)
            real_labels = labels.to(device)

            batch_size = real_images.size(0)

            # -------------------------
            # 실제 테스트 이미지
            # -------------------------
            real_output = discriminator(
                real_images,
                real_labels
            )

            real_targets = torch.ones_like(
                real_output
            )

            real_loss = criterion(
                real_output,
                real_targets
            )

            # -------------------------
            # 생성 이미지
            # -------------------------
            noise = torch.randn(
                batch_size,
                noise_dim,
                device=device
            )

            fake_images = generator(
                noise,
                real_labels
            )

            fake_output = discriminator(
                fake_images,
                real_labels
            )

            fake_targets = torch.zeros_like(
                fake_output
            )

            fake_loss = criterion(
                fake_output,
                fake_targets
            )

            # -------------------------
            # Test Loss
            # -------------------------
            loss = real_loss + fake_loss

            total_loss += loss.item()


    total_loss /= len(test_loader)

    return total_loss


# Training
for epoch in range(
    start_epoch,
    end_epoch
):

    generator.train()
    discriminator.train()

    g_epoch_loss = 0
    d_epoch_loss = 0


    for images, labels in train_loader:

        # 데이터 → Device
        real_images = images.to(device)
        real_labels = labels.to(device)

        batch_size = real_images.size(0)

        # ==================================================
        # 1. Discriminator 학습
        # ==================================================
        optimizer_D.zero_grad()


        # 실제 이미지
        real_output = discriminator(
            real_images,
            real_labels
        )

        real_targets = torch.ones_like(
            real_output
        )

        real_loss = criterion(
            real_output,
            real_targets
        )


        # 생성 이미지
        noise = torch.randn(
            batch_size,
            noise_dim,
            device=device
        )

        fake_images = generator(
            noise,
            real_labels
        )

        fake_output = discriminator(
            fake_images.detach(),
            real_labels
        )

        fake_targets = torch.zeros_like(
            fake_output
        )

        fake_loss = criterion(
            fake_output,
            fake_targets
        )

        # Discriminator Loss
        d_loss = real_loss + fake_loss

        d_loss.backward()

        optimizer_D.step()

        # ==================================================
        # 2. Generator 학습
        # ==================================================
        optimizer_G.zero_grad()

        noise = torch.randn(
            batch_size,
            noise_dim,
            device=device
        )

        fake_images = generator(
            noise,
            real_labels
        )

        fake_output = discriminator(
            fake_images,
            real_labels
        )

        generator_targets = torch.ones_like(
            fake_output
        )

        g_loss = criterion(
            fake_output,
            generator_targets
        )

        g_loss.backward()

        optimizer_G.step()

        # -------------------------
        # Epoch Loss 누적
        # -------------------------
        d_epoch_loss += d_loss.item()
        g_epoch_loss += g_loss.item()

    # =========================
    # Epoch Loss
    # =========================
    d_epoch_loss /= len(train_loader)
    g_epoch_loss /= len(train_loader)

    # =========================
    # Test
    # =========================
    test_loss = test_discriminator()

    current_epoch = epoch + 1

    print(
        f"Epoch [{current_epoch}/{end_epoch}] "
        f"D Loss: {d_epoch_loss:.4f} "
        f"G Loss: {g_epoch_loss:.4f} "
        f"Test Loss: {test_loss:.4f}"
    )

    # ==================================================
    # Checkpoint 생성
    # ==================================================
    checkpoint = {
        "epoch": current_epoch,
        "generator_state_dict": generator.state_dict(),
        "discriminator_state_dict": discriminator.state_dict(),
        "optimizer_G_state_dict": optimizer_G.state_dict(),
        "optimizer_D_state_dict": optimizer_D.state_dict()
    }

    # ==================================================
    # 1. Latest 저장
    # ==================================================
    latest_path = os.path.join(
        latest_dir,
        f"checkpoint_{current_epoch}.pth"
    )

    torch.save(
        checkpoint,
        latest_path
    )

    # ==================================================
    # 2. 최근 10개만 유지
    # ==================================================
    checkpoint_files = [
        file
        for file in os.listdir(latest_dir)
        if file.endswith(".pth")
    ]

    checkpoint_files.sort(
        key=lambda file: int(
            file.split("_")[1].split(".")[0]
        )
    )

    while len(checkpoint_files) > 10:

        oldest_file = checkpoint_files.pop(0)

        oldest_path = os.path.join(
            latest_dir,
            oldest_file
        )

        os.remove(oldest_path)

    # ==================================================
    # 3. 10 Epoch마다 Periodic 저장
    # ==================================================
    if current_epoch % 10 == 0:

        periodic_path = os.path.join(
            periodic_dir,
            f"checkpoint_{current_epoch}.pth"
        )

        torch.save(
            checkpoint,
            periodic_path
        )

        print(
            f"Periodic Checkpoint 저장: "
            f"{periodic_path}"
        )

