import os
import time
import torch
import torch.nn as nn

from dataset import train_loader, test_loader
from model import Generator, Discriminator

# 기본 설정
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

num_epochs = 20
noise_dim = 100
train_ratio = [1, 2] # Discriminator : Generator 학습 비율

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

milestone_dir = os.path.join(
    checkpoint_dir,
    "checkpoint_Milestone"
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

os.makedirs(
    milestone_dir,
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

# Test 함수
def test_discriminator():

    generator.eval()
    discriminator.eval()

    real_loss_total = 0
    wrong_loss_total = 0
    fake_loss_total = 0
    total_loss = 0

    real_output_total = 0
    wrong_output_total = 0
    fake_output_total = 0

    with torch.no_grad():

        for images, labels in test_loader:

            real_images = images.to(device)
            real_labels = labels.to(device)

            batch_size = real_images.size(0)

            # -------------------------
            # 실제 이미지 + 올바른 label
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
            # 실제 이미지 + 잘못된 label
            # -------------------------
            wrong_offsets = torch.randint(
                1,
                10,
                (batch_size,),
                device=device
            )

            wrong_labels = (
                real_labels + wrong_offsets
            ) % 10

            wrong_output = discriminator(
                real_images,
                wrong_labels
            )

            wrong_targets = torch.zeros_like(
                wrong_output
            )

            wrong_loss = criterion(
                wrong_output,
                wrong_targets
            )

            # -------------------------
            # 생성 이미지 + 올바른 label
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
            loss = (
                real_loss
                + wrong_loss
                + fake_loss
            )

            # -------------------------
            # Test 결과 누적
            # -------------------------
            real_loss_total += real_loss.item()
            wrong_loss_total += wrong_loss.item()
            fake_loss_total += fake_loss.item()
            total_loss += loss.item()

            real_output_total += real_output.mean().item()
            wrong_output_total += wrong_output.mean().item()
            fake_output_total += fake_output.mean().item()

    num_batches = len(test_loader)

    return {
        "real_loss": real_loss_total / num_batches,
        "wrong_loss": wrong_loss_total / num_batches,
        "fake_loss": fake_loss_total / num_batches,
        "total_loss": total_loss / num_batches,

        "real_output": real_output_total / num_batches,
        "wrong_output": wrong_output_total / num_batches,
        "fake_output": fake_output_total / num_batches
    }


# Checkpoint 0 저장
checkpoint = {
    "epoch": 0,
    "generator_state_dict": generator.state_dict(),
    "discriminator_state_dict": discriminator.state_dict(),
    "optimizer_G_state_dict": optimizer_G.state_dict(),
    "optimizer_D_state_dict": optimizer_D.state_dict()
}

checkpoint_path = os.path.join(
    milestone_dir,
    "checkpoint_0.pth"
)

torch.save(
    checkpoint,
    checkpoint_path
)

print(
    f"Milestone Checkpoint 저장: "
    f"{checkpoint_path}"
)

# 총 시간 기록
total_start_time = time.perf_counter()

# Training
for epoch in range(num_epochs):

    # epoch 시간 기록
    epoch_start_time = time.perf_counter()

    generator.train()
    discriminator.train()

    g_epoch_loss = 0

    d_epoch_loss = 0
    d_real_loss = 0
    d_wrong_loss = 0
    d_fake_loss = 0

    d_real_output = 0
    d_wrong_output = 0
    d_fake_output = 0

    g_fake_output = 0

    for images, labels in train_loader:

        # 데이터 → Device
        real_images = images.to(
            device,
            non_blocking=True
        )

        real_labels = labels.to(
            device,
            non_blocking=True
        )

        batch_size = real_images.size(0)

        # ==================================================
        # 1. Discriminator 학습
        # ==================================================
        for _ in range(train_ratio[0]):

            optimizer_D.zero_grad(set_to_none=True)

            # -------------------------
            # 실제 이미지 + 올바른 label
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
            # 실제 이미지 + 잘못된 label
            # -------------------------
            wrong_offsets = torch.randint(
                1,
                10,
                (batch_size,),
                device=device
            )

            wrong_labels = (
                real_labels + wrong_offsets
            ) % 10

            wrong_output = discriminator(
                real_images,
                wrong_labels
            )

            wrong_targets = torch.zeros_like(
                wrong_output
            )

            wrong_loss = criterion(
                wrong_output,
                wrong_targets
            )

            # -------------------------
            # 생성 이미지 + 올바른 label
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

            # -------------------------
            # Discriminator Loss
            # -------------------------
            d_loss = (
                real_loss
                + wrong_loss
                + fake_loss
            )

            # Gradient 계산
            d_loss.backward()

            # Discriminator 업데이트
            optimizer_D.step()

            # Epoch Loss 누적
            d_epoch_loss += d_loss.item()

            d_real_loss += real_loss.item()
            d_wrong_loss += wrong_loss.item()
            d_fake_loss += fake_loss.item()

            d_real_output += real_output.mean().item()
            d_wrong_output += wrong_output.mean().item()
            d_fake_output += fake_output.mean().item()

        # ==================================================
        # 2. Generator 학습
        # ==================================================
        # D gradient 계산 비활성화
        for param in discriminator.parameters():
            param.requires_grad = False

        for _ in range(train_ratio[1]):

            optimizer_G.zero_grad(set_to_none=True)

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

            # Epoch Loss 누적
            g_epoch_loss += g_loss.item()

            g_fake_output += fake_output.mean().item()

        # D gradient 계산 활성화
        for param in discriminator.parameters():
            param.requires_grad = True

    # =========================
    # Epoch 결과
    # =========================
    d_step_count = (
        len(train_loader) * train_ratio[0]
    )

    g_step_count = (
        len(train_loader) * train_ratio[1]
    )

    d_epoch_loss /= d_step_count

    d_real_loss /= d_step_count
    d_wrong_loss /= d_step_count
    d_fake_loss /= d_step_count

    d_real_output /= d_step_count
    d_wrong_output /= d_step_count
    d_fake_output /= d_step_count

    g_epoch_loss /= g_step_count
    g_fake_output /= g_step_count

    # Test
    # Test
    test_result = test_discriminator()

    # 시간 계산
    epoch_time = time.perf_counter() - epoch_start_time
    elapsed_time = time.perf_counter() - total_start_time

    # Learning Rate
    g_lr = optimizer_G.param_groups[0]["lr"]
    d_lr = optimizer_D.param_groups[0]["lr"]

    print(
        f"Epoch [{epoch + 1}/{num_epochs}]"
    )

    print(
        f"D Loss | "
        f"Real: {d_real_loss:.4f} "
        f"Wrong: {d_wrong_loss:.4f} "
        f"Fake: {d_fake_loss:.4f} "
        f"Total: {d_epoch_loss:.4f}"
    )

    print(
        f"D Output | "
        f"Real: {d_real_output:.4f} "
        f"Wrong: {d_wrong_output:.4f} "
        f"Fake: {d_fake_output:.4f}"
    )

    print(
        f"G Loss: {g_epoch_loss:.4f} "
        f"G Fake Output: {g_fake_output:.4f}"
    )

    print(
        f"Test Loss | "
        f"Real: {test_result['real_loss']:.4f} "
        f"Wrong: {test_result['wrong_loss']:.4f} "
        f"Fake: {test_result['fake_loss']:.4f} "
        f"Total: {test_result['total_loss']:.4f}"
    )

    print(
        f"Test Output | "
        f"Real: {test_result['real_output']:.4f} "
        f"Wrong: {test_result['wrong_output']:.4f} "
        f"Fake: {test_result['fake_output']:.4f}"
    )

    print(
        f"Learning Rate | "
        f"G: {g_lr:.6f} "
        f"D: {d_lr:.6f}"
    )

    print(
        f"Epoch Time: {epoch_time:.2f}s "
        f"Elapsed Time: {elapsed_time / 60:.2f}min"
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

    # 1. Latest checkpoint
    latest_path = os.path.join(
        latest_dir,
        f"checkpoint_{current_epoch}.pth"
    )

    torch.save(
        checkpoint,
        latest_path
    )

    # 2. 최근 10개만 유지
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

    # 3. 10 epoch마다 Periodic checkpoint 저장
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

    # 4. 10 이하 Milestone checkpoint 저장
    if current_epoch < 10:

        milestone_path = os.path.join(
            milestone_dir,
            f"checkpoint_{current_epoch}.pth"
        )

        torch.save(
            checkpoint,
            milestone_path
        )

        print(
            f"Milestone Checkpoint 저장: "
            f"{milestone_path}"
        )

