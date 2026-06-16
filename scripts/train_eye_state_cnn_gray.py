import os
import random
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageFile
from sklearn.metrics import confusion_matrix, classification_report

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm


ImageFile.LOAD_TRUNCATED_IMAGES = True


# ==========================================================
# Config
# ==========================================================
@dataclass
class Config:
    # data
    data_root: str = r"data/raw/eyes_set"
    train_dir: str = "train"
    val_dir: str = "val"
    test_dir: str = "test"

    # system
    seed: int = 42
    num_workers: int = 12
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # image
    image_size: int = 224
    input_channels: int = 1

    # grayscale normalize
    gray_mean: float = 0.5
    gray_std: float = 0.5

    # training
    model_name: str = "mobilenetv4_conv_small.e2400_r224_in1k"
    pretrained: bool = True
    epochs: int = 50
    batch_size: int = 64
    lr: float = 1e-5
    weight_decay: float = 1e-4
    label_smoothing: float = 0.0

    # 기존 3채널 checkpoint를 1채널 모델로 변환해서 이어 학습할 때 사용
    # 기존 best_eye_model.pth 경로로 맞춰라.
    rgb_checkpoint_path: str = r""

    # finetune
    freeze_backbone: bool = False

    # early stop
    use_early_stopping: bool = False
    patience: int = 4

    # save
    save_dir: str = r"outputs/eye_state_cnn"
    best_ckpt_name: str = "best_eye_model_gray1.pth"
    last_ckpt_name: str = "last_eye_model_gray1.pth"


CFG = Config()


# ==========================================================
# Utils
# ==========================================================
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def save_config(cfg: Config, save_dir: str) -> None:
    ensure_dir(save_dir)
    with open(os.path.join(save_dir, "config.txt"), "w", encoding="utf-8") as f:
        for k, v in asdict(cfg).items():
            f.write(f"{k}: {v}\n")


def gray_pil_loader(path: str):
    """
    ImageFolder 기본 loader는 RGB로 바꾸는 경우가 많다.
    그래서 여기서는 강제로 L 모드, 즉 grayscale 1채널로 읽는다.
    """
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("L")


def get_transforms(image_size: int, mean: float, std: float):
    train_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=8),

        # grayscale에서도 brightness/contrast는 사용 가능
        transforms.ColorJitter(brightness=0.2, contrast=0.2),

        transforms.ToTensor(),  # [1, H, W]
        transforms.Normalize(mean=[mean], std=[std]),
    ])

    eval_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),  # [1, H, W]
        transforms.Normalize(mean=[mean], std=[std]),
    ])

    return train_tf, eval_tf


def build_model(
    model_name: str,
    num_classes: int,
    pretrained: bool,
    freeze_backbone: bool,
    in_chans: int = 1,
) -> nn.Module:
    """
    timm 모델을 1채널 입력으로 생성.
    pretrained=True + in_chans=1이면 timm이 ImageNet RGB weight를 1채널용으로 자동 변환한다.
    """
    model = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=num_classes,
        in_chans=in_chans,
    )

    if freeze_backbone:
        for name, param in model.named_parameters():
            if "classifier" not in name and "head" not in name:
                param.requires_grad = False

    return model


def safe_torch_load(path: str, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[len("module."):]] = v
        else:
            new_state_dict[k] = v
    return new_state_dict


def adapt_rgb_checkpoint_to_gray_model(
    model: nn.Module,
    checkpoint_path: str,
    device: str,
) -> None:
    """
    기존 3채널 모델 checkpoint를 1채널 모델에 로드한다.

    핵심:
    기존 첫 Conv weight: [out, 3, k, k]
    변경 첫 Conv weight: [out, 1, k, k]

    grayscale 이미지를 RGB 3채널로 복제해서 넣던 효과와 비슷하게 만들기 위해
    RGB 방향 weight를 sum해서 1채널 weight로 만든다.
    """
    print(f"[INFO] Loading RGB checkpoint and adapting to gray model:")
    print(f"       {checkpoint_path}")

    checkpoint = safe_torch_load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        source_state = checkpoint["model_state_dict"]
    else:
        source_state = checkpoint

    source_state = strip_module_prefix(source_state)
    target_state = model.state_dict()

    adapted_state = {}
    adapted_keys = []
    skipped_keys = []

    for key, src_tensor in source_state.items():
        if key not in target_state:
            skipped_keys.append((key, "not in target model"))
            continue

        tgt_tensor = target_state[key]

        # shape이 완전히 같으면 그대로 로드
        if src_tensor.shape == tgt_tensor.shape:
            adapted_state[key] = src_tensor
            continue

        # RGB Conv weight [out, 3, k, k] -> Gray Conv weight [out, 1, k, k]
        if (
            src_tensor.ndim == 4
            and tgt_tensor.ndim == 4
            and src_tensor.shape[1] == 3
            and tgt_tensor.shape[1] == 1
            and src_tensor.shape[0] == tgt_tensor.shape[0]
            and src_tensor.shape[2:] == tgt_tensor.shape[2:]
        ):
            adapted_state[key] = src_tensor.sum(dim=1, keepdim=True)
            adapted_keys.append(key)
            continue

        # classifier 클래스 수가 다르거나 구조가 다르면 스킵
        skipped_keys.append((key, f"shape mismatch {tuple(src_tensor.shape)} -> {tuple(tgt_tensor.shape)}"))

    incompatible = model.load_state_dict(adapted_state, strict=False)

    print(f"[INFO] Loaded tensors: {len(adapted_state)}")
    print(f"[INFO] Adapted RGB->Gray conv tensors: {adapted_keys}")

    if len(skipped_keys) > 0:
        print(f"[WARN] Skipped tensors: {len(skipped_keys)}")
        for k, reason in skipped_keys[:10]:
            print(f"       - {k}: {reason}")
        if len(skipped_keys) > 10:
            print("       ...")

    if len(incompatible.missing_keys) > 0:
        print(f"[WARN] Missing keys after load: {len(incompatible.missing_keys)}")
        for k in incompatible.missing_keys[:10]:
            print(f"       - {k}")
        if len(incompatible.missing_keys) > 10:
            print("       ...")

    if len(incompatible.unexpected_keys) > 0:
        print(f"[WARN] Unexpected keys after load: {len(incompatible.unexpected_keys)}")
        for k in incompatible.unexpected_keys[:10]:
            print(f"       - {k}")
        if len(incompatible.unexpected_keys) > 10:
            print("       ...")


def print_first_conv_shape(model: nn.Module) -> None:
    for name, param in model.named_parameters():
        if param.ndim == 4:
            print(f"[CHECK] First conv-like weight: {name} {tuple(param.shape)}")
            return
    print("[WARN] Conv weight를 찾지 못했습니다.")


def get_dataloaders(cfg: Config):
    train_tf, eval_tf = get_transforms(
        image_size=cfg.image_size,
        mean=cfg.gray_mean,
        std=cfg.gray_std,
    )

    train_path = os.path.join(cfg.data_root, cfg.train_dir)
    val_path = os.path.join(cfg.data_root, cfg.val_dir)
    test_path = os.path.join(cfg.data_root, cfg.test_dir)

    train_ds = datasets.ImageFolder(
        train_path,
        transform=train_tf,
        loader=gray_pil_loader,
    )
    val_ds = datasets.ImageFolder(
        val_path,
        transform=eval_tf,
        loader=gray_pil_loader,
    )
    test_ds = datasets.ImageFolder(
        test_path,
        transform=eval_tf,
        loader=gray_pil_loader,
    )

    class_to_idx = train_ds.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    if val_ds.class_to_idx != class_to_idx or test_ds.class_to_idx != class_to_idx:
        raise ValueError("train/val/test class mapping mismatch. 폴더 이름을 동일하게 맞춰야 합니다.")

    pin_memory = cfg.device == "cuda"

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=pin_memory,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader, idx_to_class


def check_input_shape(loader: DataLoader, cfg: Config) -> None:
    images, targets = next(iter(loader))
    print(f"[CHECK] Batch image shape: {tuple(images.shape)}")
    print(f"[CHECK] Batch target shape: {tuple(targets.shape)}")

    if images.ndim != 4:
        raise ValueError(f"입력 텐서가 4차원이 아닙니다: {tuple(images.shape)}")

    if images.shape[1] != cfg.input_channels:
        raise ValueError(
            f"입력 채널 수가 {cfg.input_channels}이 아닙니다. "
            f"현재 shape: {tuple(images.shape)}"
        )

    print("[CHECK] 입력 데이터는 정상적으로 1채널입니다.")


# ==========================================================
# Train / Eval
# ==========================================================
@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion, device: str):
    model.eval()

    running_loss = 0.0
    running_correct = 0
    total = 0

    all_preds = []
    all_targets = []

    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)

        logits = model(images)
        loss = criterion(logits, targets)

        running_loss += loss.item() * images.size(0)

        preds = torch.argmax(logits, dim=1)
        running_correct += (preds == targets).sum().item()
        total += targets.size(0)

        all_preds.extend(preds.cpu().numpy().tolist())
        all_targets.extend(targets.cpu().numpy().tolist())

    avg_loss = running_loss / total
    avg_acc = running_correct / total

    return avg_loss, avg_acc, np.array(all_targets), np.array(all_preds)


def train_one_epoch(model, loader, criterion, optimizer, device: str):
    model.train()

    running_loss = 0.0
    running_correct = 0
    total = 0

    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        logits = model(images)
        loss = criterion(logits, targets)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        preds = torch.argmax(logits, dim=1)
        running_correct += (preds == targets).sum().item()
        total += targets.size(0)

    avg_loss = running_loss / total
    avg_acc = running_correct / total

    return avg_loss, avg_acc


# ==========================================================
# Plot / Save
# ==========================================================
def plot_losses(history: Dict[str, List[float]], save_dir: str) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(history["train_loss"], label="train_loss")
    plt.plot(history["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train / Validation Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "loss_curve.png"), dpi=200)
    plt.close()


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    save_path: str,
    title: str = "Confusion Matrix",
    normalize: bool = False,
) -> None:
    if normalize:
        cm_to_show = cm.astype(np.float64) / np.clip(cm.sum(axis=1, keepdims=True), a_min=1, a_max=None)
    else:
        cm_to_show = cm

    plt.figure(figsize=(6, 5))
    plt.imshow(cm_to_show, interpolation="nearest")
    plt.title(title)
    plt.colorbar()

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    threshold = cm_to_show.max() / 2.0 if cm_to_show.max() > 0 else 0.5

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            text_value = f"{cm_to_show[i, j]:.2f}" if normalize else f"{cm[i, j]:d}"
            plt.text(
                j,
                i,
                text_value,
                ha="center",
                va="center",
                color="white" if cm_to_show[i, j] > threshold else "black",
            )

    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def save_classification_report(y_true, y_pred, class_names, save_dir: str):
    report_str = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4,
        zero_division=0,
    )

    report_path = os.path.join(save_dir, "classification_report.txt")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_str)

    return report_str


def save_checkpoint(
    path: str,
    epoch: int,
    model: nn.Module,
    optimizer: optim.Optimizer,
    val_loss: float,
    class_names: List[str],
    cfg: Config,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss,
            "class_names": class_names,
            "config": asdict(cfg),
            "input_channels": cfg.input_channels,
            "gray_mean": cfg.gray_mean,
            "gray_std": cfg.gray_std,
        },
        path,
    )


# ==========================================================
# Main
# ==========================================================
def main(cfg: Config):
    set_seed(cfg.seed)
    ensure_dir(cfg.save_dir)
    save_config(cfg, cfg.save_dir)

    print("=" * 60)
    print("Config")
    for k, v in asdict(cfg).items():
        print(f"{k}: {v}")
    print("=" * 60)

    train_loader, val_loader, test_loader, idx_to_class = get_dataloaders(cfg)

    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]

    print(f"Classes: {class_names}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")

    # 진짜 1채널로 들어가는지 확인
    check_input_shape(train_loader, cfg)

    # 기존 RGB checkpoint가 있으면, ImageNet pretrained를 다시 받을 필요 없이 checkpoint를 변환해서 사용
    has_rgb_checkpoint = (
        cfg.rgb_checkpoint_path is not None
        and str(cfg.rgb_checkpoint_path).strip() != ""
        and os.path.isfile(cfg.rgb_checkpoint_path)
    )

    if cfg.rgb_checkpoint_path and not os.path.isfile(cfg.rgb_checkpoint_path):
        print(f"[WARN] rgb_checkpoint_path 파일이 없습니다:")
        print(f"       {cfg.rgb_checkpoint_path}")
        print("[WARN] 기존 checkpoint 없이 ImageNet pretrained 1채널 모델로 시작합니다.")

    model = build_model(
        model_name=cfg.model_name,
        num_classes=len(class_names),
        pretrained=cfg.pretrained and not has_rgb_checkpoint,
        freeze_backbone=cfg.freeze_backbone,
        in_chans=cfg.input_channels,
    ).to(cfg.device)

    print_first_conv_shape(model)

    # 기존 3채널 best_eye_model.pth를 1채널 모델로 변환해서 로드
    if has_rgb_checkpoint:
        adapt_rgb_checkpoint_to_gray_model(
            model=model,
            checkpoint_path=cfg.rgb_checkpoint_path,
            device=cfg.device,
        )
        print("[INFO] RGB checkpoint -> 1-channel model 변환 로드 완료.")
        print_first_conv_shape(model)

    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
    )

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    best_val_loss = float("inf")
    best_epoch = -1
    early_stop_counter = 0

    best_ckpt_path = os.path.join(cfg.save_dir, cfg.best_ckpt_name)
    last_ckpt_path = os.path.join(cfg.save_dir, cfg.last_ckpt_name)

    for epoch in range(cfg.epochs):
        start_time = time.time()

        train_loss, train_acc = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=cfg.device,
        )

        val_loss, val_acc, _, _ = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=cfg.device,
        )

        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        elapsed = time.time() - start_time

        print(
            f"[Epoch {epoch + 1:02d}/{cfg.epochs:02d}] "
            f"Train Loss={train_loss:.4f} Acc={train_acc:.4f} | "
            f"Val Loss={val_loss:.4f} Acc={val_acc:.4f} | "
            f"Time={elapsed:.2f}s"
        )

        save_checkpoint(
            path=last_ckpt_path,
            epoch=epoch + 1,
            model=model,
            optimizer=optimizer,
            val_loss=val_loss,
            class_names=class_names,
            cfg=cfg,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            early_stop_counter = 0

            save_checkpoint(
                path=best_ckpt_path,
                epoch=epoch + 1,
                model=model,
                optimizer=optimizer,
                val_loss=val_loss,
                class_names=class_names,
                cfg=cfg,
            )

            print(f"[INFO] Best checkpoint saved -> {best_ckpt_path}")
        else:
            early_stop_counter += 1

        if cfg.use_early_stopping and early_stop_counter >= cfg.patience:
            print("[INFO] Early stopping triggered.")
            break

    print("\n" + "=" * 60)
    print(f"Best epoch: {best_epoch}")
    print(f"Best val loss: {best_val_loss:.4f}")
    print("=" * 60)

    # best 모델 로드 후 test 평가
    checkpoint = safe_torch_load(best_ckpt_path, map_location=cfg.device)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_loss, test_acc, y_true, y_pred = evaluate(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=cfg.device,
    )

    print(f"\n[Test] Loss={test_loss:.4f} Acc={test_acc:.4f}")

    report_str = save_classification_report(
        y_true=y_true,
        y_pred=y_pred,
        class_names=class_names,
        save_dir=cfg.save_dir,
    )

    print("\nClassification Report")
    print(report_str)

    cm = confusion_matrix(y_true, y_pred)

    np.savetxt(
        os.path.join(cfg.save_dir, "test_confusion_matrix.csv"),
        cm,
        delimiter=",",
        fmt="%d",
    )

    plot_confusion_matrix(
        cm=cm,
        class_names=class_names,
        save_path=os.path.join(cfg.save_dir, "test_confusion_matrix.png"),
        title="Test Confusion Matrix",
        normalize=False,
    )

    plot_confusion_matrix(
        cm=cm,
        class_names=class_names,
        save_path=os.path.join(cfg.save_dir, "test_confusion_matrix_normalized.png"),
        title="Test Confusion Matrix (Normalized)",
        normalize=True,
    )

    plot_losses(history, cfg.save_dir)

    print(f"\n[INFO] Results saved to: {cfg.save_dir}")
    print(f"[INFO] Best checkpoint: {best_ckpt_path}")
    print(f"[INFO] Last checkpoint: {last_ckpt_path}")
    print(f"[INFO] Saved: test_confusion_matrix.png")
    print(f"[INFO] Saved: test_confusion_matrix_normalized.png")
    print(f"[INFO] Saved: classification_report.txt")


if __name__ == "__main__":
    main(CFG)