import torch
from trainer import Trainer
from utils.config import cfg
from datasets.data import get_semi_aves
import torchvision.utils as vutils

def dry_run_debug():
    cfg.merge_from_file("configs/peft/semi_aves.yaml")
    cfg.freeze()

    # Load datasets
    train_labeled_dataset, train_unlabeled_dataset, test_dataset = get_semi_aves(cfg)
    print(f"[DEBUG] Labeled dataset size: {len(train_labeled_dataset)}", flush=True)
    print(f"[DEBUG] Unlabeled dataset size: {len(train_unlabeled_dataset)}", flush=True)
    print(f"[DEBUG] Test dataset size: {len(test_dataset)}", flush=True)

    for i in range(3):
        img, label, _ = train_labeled_dataset[i]
        print(f"[DEBUG] Sample {i}: label = {label}", flush=True)

    # Setup trainer and model
    trainer = Trainer(cfg)
    trainer.build_data_loader()
    trainer.build_model()

    # Get a single batch
    loader = iter(trainer.train_loader)
    images, labels, _ = next(loader)
    batch_imgs = images[0] if isinstance(images, tuple) else images

    print(f"[DEBUG] One batch images shape: {batch_imgs.shape}", flush=True)
    print(f"[DEBUG] Labels: {labels[:10]}", flush=True)

    vutils.save_image(batch_imgs[:16], 'debug_batch.png', normalize=True, nrow=4)
    print("[DEBUG] Saved batch preview to debug_batch.png", flush=True)

    # Overfit one batch
    print("[DEBUG] Starting overfit on single batch...", flush=True)
    model = trainer.model
    model.train()
    device = trainer.device
    model.to(device)
    images = batch_imgs.to(device)
    labels = labels.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    loss_fn = torch.nn.CrossEntropyLoss()

    for i in range(100):
        optimizer.zero_grad()
        logits = model(images)
        loss = loss_fn(logits, labels)
        acc = (logits.argmax(dim=1) == labels).float().mean().item()
        loss.backward()
        optimizer.step()
        print(f"[OVERFIT] Iter {i:03d} - Loss: {loss.item():.4f}, Acc: {acc:.4f}", flush=True)

if __name__ == "__main__":
    dry_run_debug()
