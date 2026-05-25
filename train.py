"""
M2-CL training script.

Usage:
  python train.py --dataset pacs --test_domain photo
  python train.py --dataset vlcs --test_domain Caltech101 --epochs 30
"""
import argparse
import os
import sys
import yaml
import torch
import torch.optim as optim
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from models import M2CL
from losses import MultiLayerContrastiveLoss


def accuracy(output, target):
    pred = output.argmax(dim=1)
    return (pred == target).float().mean().item()


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, total_acc = 0.0, 0.0
    for imgs, labels in tqdm(loader, leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits, embeddings = model(imgs)
        loss = criterion(logits, labels, embeddings)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_acc += accuracy(logits, labels)
    n = len(loader)
    return total_loss / n, total_acc / n


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits, _ = model(imgs)
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return correct / total if total > 0 else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True,
                        choices=["pacs", "vlcs", "office_home", "nico"])
    parser.add_argument("--test_domain", required=True)
    parser.add_argument("--data_root", default="./data_root")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--save_dir", default="./outputs/checkpoints")
    args = parser.parse_args()

    cfg_path = os.path.join(os.path.dirname(__file__), "configs", f"{args.dataset}.yaml")
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    epochs = args.epochs or cfg["epochs"]
    batch_size = args.batch_size or cfg["batch_size"]
    lr = args.lr or cfg["lr"]
    alpha = args.alpha or cfg["alpha"]
    temperature = args.temperature or cfg["temperature"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Dataset: {args.dataset} | Test domain: {args.test_domain}")

    if args.dataset == "pacs":
        from data import get_pacs_loaders
        train_loader, test_loader = get_pacs_loaders(
            args.data_root, args.test_domain, batch_size)
    elif args.dataset == "vlcs":
        from data import get_vlcs_loaders
        train_loader, test_loader = get_vlcs_loaders(
            args.data_root, args.test_domain, batch_size)
    elif args.dataset == "office_home":
        from data import get_office_home_loaders
        train_loader, test_loader = get_office_home_loaders(
            args.data_root, args.test_domain, batch_size)
    else:
        raise ValueError("NICO requires special context setup — see evaluate.py")

    model = M2CL(num_classes=cfg["num_classes"]).to(device)
    criterion = MultiLayerContrastiveLoss(alpha=alpha, temperature=temperature)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    os.makedirs(args.save_dir, exist_ok=True)
    best_acc = 0.0

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device)
        test_acc = evaluate(model, test_loader, device)
        scheduler.step()

        print(f"Epoch {epoch:3d}/{epochs} | "
              f"loss={train_loss:.4f} | train_acc={train_acc:.4f} | "
              f"test_acc={test_acc:.4f}")

        if test_acc > best_acc:
            best_acc = test_acc
            ckpt = os.path.join(
                args.save_dir,
                f"{args.dataset}_{args.test_domain}_best.pth"
            )
            torch.save({"epoch": epoch, "state_dict": model.state_dict(),
                        "acc": best_acc}, ckpt)

    print(f"\nBest test accuracy on {args.test_domain}: {best_acc:.4f}")


if __name__ == "__main__":
    main()
