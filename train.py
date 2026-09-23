import os
import torch
import torch.nn as nn

from dataset import train_loader
from model import Generator, Discriminator

# 기본 설정
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

num_epochs = 25
noise_dim = 100

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

# 폴더가 없으면 생성
os.makedirs(
    periodic_dir,
    exist_ok=True
)

os.makedirs(
    latest_dir,
    exist_ok=True
)

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

# Training
for epoch in range(num_epochs):

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

        # Gradient 계산
        d_loss.backward()

        # Discriminator 업데이트
        optimizer_D.step()

        # ==================================================
        # 2. Generator 학습
        # ==================================================
        optimizer_G.zero_grad()

        # 새로운 Noise 생성
        noise = torch.randn(
            batch_size,
            noise_dim,
            device=device
        )

        fake_images = generator(
            noise,
            real_labels
        )

        # Discriminator를 통과
        fake_output = discriminator(
            fake_images,
            real_labels
        )

        # Generator는 Fake를 Real로 판별하도록 학습
        generator_targets = torch.ones_like(
            fake_output
        )

        g_loss = criterion(
            fake_output,
            generator_targets
        )

        # Gradient 계산
        g_loss.backward()

        # Generator 업데이트
        optimizer_G.step()

        # -------------------------
        # Epoch Loss 누적
        # -------------------------

        d_epoch_loss += d_loss.item()
        g_epoch_loss += g_loss.item()

    # =========================
    # Epoch 결과
    # =========================

    d_epoch_loss /= len(train_loader)
    g_epoch_loss /= len(train_loader)

    print(
        f"Epoch [{epoch + 1}/{num_epochs}] "
        f"D Loss: {d_epoch_loss:.4f} "
        f"G Loss: {g_epoch_loss:.4f}"
    )

    # ==================================================
    # Checkpoint 저장
    # ==================================================

    current_epoch = epoch + 1

    checkpoint = {
        "epoch": current_epoch,
        "generator_state_dict": generator.state_dict(),
        "discriminator_state_dict": discriminator.state_dict(),
        "optimizer_G_state_dict": optimizer_G.state_dict(),
        "optimizer_D_state_dict": optimizer_D.state_dict()
    }

    # --------------------------------------------------
    # 1. Latest checkpoint
    # --------------------------------------------------

    latest_path = os.path.join(
        latest_dir,
        f"checkpoint_{current_epoch}.pth"
    )

    torch.save(
        checkpoint,
        latest_path
    )

    # --------------------------------------------------
    # 2. 최근 10개만 유지
    # --------------------------------------------------

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

    # --------------------------------------------------
    # 3. 10 epoch마다 Periodic checkpoint 저장
    # --------------------------------------------------

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
            f"Checkpoint 저장: "
            f"{periodic_path}"
        )
