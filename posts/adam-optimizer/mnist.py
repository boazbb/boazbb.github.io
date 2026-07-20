
import json
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# ── Mode: "train" (full run) or "grid" (LR search, 5 epochs) ─────────────────
MODE = "train"

CACHE_PATH      = "./data/mnist_results.json"
GRID_CACHE_PATH = "./data/mnist_grid_results.json"
EPOCHS      = 15
GRID_EPOCHS = 5
N_SEEDS     = 5

plt.style.use("seaborn-v0_8-darkgrid")

# ── Tiny CNN ──────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 64), nn.ReLU(),
            nn.Linear(64, 10),
        )
    def forward(self, x):
        return self.net(x)

# ── Data ──────────────────────────────────────────────────────────────────────

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_data = datasets.MNIST("./data", train=True, download=True, transform=transform)
train_loader = DataLoader(train_data, batch_size=512, shuffle=True, num_workers=0)

# ── Training loop ─────────────────────────────────────────────────────────────

def train(optimizer_cls, opt_kwargs, epochs=EPOCHS, label="", seed=42):
    """Train SmallCNN with a given seed; return per-epoch average loss list."""
    torch.manual_seed(seed)
    model = SmallCNN().to(device)
    optimizer = optimizer_cls(model.parameters(), **opt_kwargs)
    criterion = nn.CrossEntropyLoss()

    epoch_losses = []
    for epoch in range(epochs):
        running = 0.0
        n = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            running += loss.item() * images.size(0)
            n += images.size(0)
        epoch_losses.append(running / n)
        print(f"  {label} seed={seed} epoch {epoch+1}/{epochs}: loss={epoch_losses[-1]:.4f}")
    return epoch_losses

# ── Optimizer configs ─────────────────────────────────────────────────────────

configs = {
    "SGD":                   (torch.optim.SGD,      dict(lr=0.1)),
    "SGD + Momentum":        (torch.optim.SGD,      dict(lr=0.01, momentum=0.9)),
    "Adagrad":               (torch.optim.Adagrad,  dict(lr=0.01)),
    "RMSProp":               (torch.optim.RMSprop,  dict(lr=0.001)),
    "RMSProp + Momentum":    (torch.optim.RMSprop,  dict(lr=0.001, momentum=0.9)),
    "Adam":                  (torch.optim.Adam,     dict(lr=0.01)),
}

search_configs = {
    "SGD":              (torch.optim.SGD,      [{"lr": lr} for lr in [0.001, 0.01, 0.1]]),
    "SGD + Momentum":   (torch.optim.SGD,      [{"lr": lr, "momentum": 0.9} for lr in [0.001, 0.01, 0.1]]),
    "AdaGrad":          (torch.optim.Adagrad,  [{"lr": lr} for lr in [0.001, 0.01, 0.1]]),
    "RMSProp":          (torch.optim.RMSprop,  [{"lr": lr} for lr in [0.0001, 0.001, 0.01]]),
    "RMSProp+Momentum": (torch.optim.RMSprop,  [{"lr": lr, "momentum": 0.9} for lr in [0.0001, 0.001, 0.01]]),
    "Adam":             (torch.optim.Adam,     [{"lr": lr} for lr in [0.0001, 0.001, 0.01]]),
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def _fill_seeds(runs, train_fn, n=N_SEEDS):
    """Append missing seed runs until len(runs) == n. Mutates runs in place."""
    while len(runs) < n:
        seed = len(runs) + 236
        runs.append(train_fn(seed=seed))
    return runs

def _plot_mean_std(ax, runs, xs, label, color):
    arr  = np.array(runs)           # (N_SEEDS, epochs)
    mean = arr.mean(axis=0)
    std  = arr.std(axis=0)
    ax.plot(xs, mean, marker="o", markersize=4, linewidth=2, label=label, color=color)
    ax.fill_between(xs, mean - std, mean + std, alpha=0.15, color=color)

# ── Importable plot functions ─────────────────────────────────────────────────

def plot_results(cache_path=CACHE_PATH, n_seeds=N_SEEDS, epochs=EPOCHS):
    """Load cached results and return a mean±std figure. Safe to import."""
    with open(cache_path) as f:
        results = json.load(f)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    xs = list(range(1, epochs + 1))
    fig, ax = plt.subplots(figsize=(8, 4))
    for (name, runs), color in zip(results.items(), colors):
        _plot_mean_std(ax, runs, xs, name, color)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training Loss")
    ax.set_yscale("log")
    ax.set_xticks(xs)
    ax.legend()
    ax.set_title(f"MNIST — Small CNN  (mean ± 1 std, {n_seeds} seeds)")
    plt.close(fig)
    return fig


def plot_grid_results(cache_path=GRID_CACHE_PATH, n_seeds=N_SEEDS, grid_epochs=GRID_EPOCHS):
    """Load cached grid results and return a mean±std subplot figure. Safe to import."""
    with open(cache_path) as f:
        grid_cache = json.load(f)
    prop_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    xs = list(range(1, grid_epochs + 1))
    n_opts = len(grid_cache)
    fig, axes = plt.subplots(2, n_opts // 2, figsize=(14, 6), sharey=True, layout="constrained")
    for ax, (name, entries) in zip(axes.flat, grid_cache.items()):
        for entry, color in zip(entries, prop_cycle):
            lr_label = f"lr={entry['kwargs']['lr']}"
            _plot_mean_std(ax, entry["runs"], xs, lr_label, color)
        ax.set_title(name, fontsize=10)
        ax.set_yscale("log")
        ax.set_xticks(xs)
        ax.legend(fontsize=8)
    fig.supxlabel("Epoch")
    fig.supylabel("Training Loss")
    fig.suptitle(f"LR Grid Search — {grid_epochs} epochs  (mean ± 1 std, {n_seeds} seeds)", fontsize=12)
    plt.close(fig)
    return fig


# ── Train mode ────────────────────────────────────────────────────────────────

if __name__ == "__main__" and MODE == "train":
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    results = json.load(open(CACHE_PATH)) if os.path.exists(CACHE_PATH) else {}

    # Handle old single-run format: list-of-floats → list-of-lists
    for name in list(results):
        if results[name] and isinstance(results[name][0], float):
            results[name] = [results[name]]

    changed = False
    for name, (cls, kwargs) in configs.items():
        runs = results.setdefault(name, [])
        before = len(runs)
        _fill_seeds(runs, lambda seed, c=cls, kw=kwargs, n=name: train(c, kw, label=n, seed=seed))
        if len(runs) > before:
            changed = True
    if changed:
        with open(CACHE_PATH, "w") as f:
            json.dump(results, f)
    print(f"Results ready ({N_SEEDS} seeds each).")

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    xs = range(1, EPOCHS + 1)
    fig, ax = plt.subplots(figsize=(8, 4))
    for (name, runs), color in zip(results.items(), colors):
        _plot_mean_std(ax, runs, xs, name, color)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training Loss")
    ax.set_yscale("log")
    ax.set_xticks(list(xs))
    ax.legend()
    ax.set_title(f"MNIST — Small CNN  (mean ± 1 std, {N_SEEDS} seeds)")
    plt.tight_layout()
    plt.show()

# ── Grid mode ─────────────────────────────────────────────────────────────────

elif __name__ == "__main__" and MODE == "grid":
    os.makedirs(os.path.dirname(GRID_CACHE_PATH), exist_ok=True)
    grid_cache = json.load(open(GRID_CACHE_PATH)) if os.path.exists(GRID_CACHE_PATH) else {}

    changed = False
    for name, (cls, kwarg_list) in search_configs.items():
        stored = grid_cache.setdefault(name, [])
        stored_by_kw = {json.dumps(r["kwargs"], sort_keys=True): r for r in stored}
        for kwargs in kwarg_list:
            key = json.dumps(kwargs, sort_keys=True)
            entry = stored_by_kw.setdefault(key, {"kwargs": kwargs, "runs": []})
            if entry not in stored:
                stored.append(entry)
            # Handle old single-run format
            if entry["runs"] and isinstance(entry["runs"][0], float):
                entry["runs"] = [entry["runs"]]
            before = len(entry["runs"])
            _fill_seeds(
                entry["runs"],
                lambda seed, c=cls, kw=kwargs, n=name: train(
                    c, kw, epochs=GRID_EPOCHS, label=f"{n} {kw}", seed=seed
                ),
            )
            if len(entry["runs"]) > before:
                changed = True

    if changed:
        with open(GRID_CACHE_PATH, "w") as f:
            json.dump(grid_cache, f)
        print("Grid results saved to cache.")
    else:
        print("Loaded grid results from cache.")

    # Summary: best LR per optimizer (by mean final-epoch loss)
    best = {}
    print(f"\n── Grid search summary ({N_SEEDS} seeds) ────────────────────────────")
    for name, (cls, _) in search_configs.items():
        best_entry = min(
            grid_cache[name],
            key=lambda r: np.mean([run[-1] for run in r["runs"]]),
        )
        mean_loss = np.mean([run[-1] for run in best_entry["runs"]])
        best[name] = (cls, best_entry["kwargs"])
        print(f"  {name}: best={best_entry['kwargs']},  mean_final_loss={mean_loss:.4f}")

    # Plot: one subplot per optimizer, all LRs overlaid (mean ± std)
    prop_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    xs = range(1, GRID_EPOCHS + 1)
    n_opts = len(search_configs)
    fig, axes = plt.subplots(2, n_opts // 2, figsize=(14, 6), sharey=True, layout="constrained")
    for ax, (name, (cls, _)) in zip(axes.flat, search_configs.items()):
        for entry, color in zip(grid_cache[name], prop_cycle):
            lr_label = f"lr={entry['kwargs']['lr']}"
            _plot_mean_std(ax, entry["runs"], xs, lr_label, color)
        ax.set_title(name, fontsize=10)
        ax.set_yscale("log")
        ax.set_xticks(list(xs))
        ax.legend(fontsize=8)
    fig.supxlabel("Epoch")
    fig.supylabel("Training Loss")
    fig.suptitle(f"LR Grid Search — {GRID_EPOCHS} epochs  (mean ± 1 std, {N_SEEDS} seeds)", fontsize=12)
    plt.show()
