import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data as data
import torchvision
import torchvision.transforms as tt
from torchvision.datasets import CelebA, CIFAR10
from tqdm import tqdm

from realnvp import RealNVP


def train(args):
    # Use cuda.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set the random seeds.
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True

    if args.dataset == "CelebA":
        # Create a dataloader for the CelebA dataset.
        transform = tt.Compose([
            tt.CenterCrop(size=148),
            tt.Resize(size=32),
            tt.PILToTensor(),
        ])

        celebs_train = CelebA("datasets", split="train", download=True, transform=transform)
        celebs_test = CelebA("datasets", split="test", download=True, transform=transform)
        train_loader = data.DataLoader(celebs_train, batch_size=args.batch_size, shuffle=True)
        test_loader = data.DataLoader(celebs_test, batch_size=args.batch_size)

    elif args.dataset == "CIFAR10":
        # transform = tt.Compose([
        #     tt.RandomHorizontalFlip(),
        #     tt.PILToTensor(),
        # ])

        # Create a dataloader for the CIFAR-10 dataset.
        cifar10_train = CIFAR10("datasets", train=True, download=True, transform=tt.PILToTensor())
        cifar10_test = CIFAR10("datasets", train=False, download=True, transform=tt.PILToTensor())
        train_loader = data.DataLoader(cifar10_train, batch_size=args.batch_size, shuffle=True)
        test_loader = data.DataLoader(cifar10_test, batch_size=args.batch_size)

    else:
        raise NotImplementedError

    # Initialize the model.
    C, H, W = (3, 32, 32)
    n_colors = 256
    model = RealNVP(in_shape=(C, H, W), n_colors=n_colors)
    model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Run the training loop.
    train_losses, test_losses = [], []
    total_norms = []
    test_losses.append(eval(model, test_loader))
    for i in tqdm(range(args.epochs)):
        avg_loss, j = 0., 0

        # Iterate over the training set.
        for x, _ in train_loader:
            # Forward pass.
            log_prob = model.log_prob(x)
            loss = -torch.mean(log_prob) / (C * H * W) # divide by the number of dims to improve training

            # Backward pass.
            optimizer.zero_grad()
            loss.backward()
            total_norm = torch.norm(torch.stack([torch.norm(p.grad) for p in model.parameters()]))
            if args.clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            optimizer.step()

            total_norms.append(total_norm.item())
            train_losses.append(loss.item())
            avg_loss += loss.item()
            j += 1

        avg_loss /= j

        # Test on the test set.
        test_losses.append(eval(model, test_loader))

        # Maybe printout results.
        if args.verbose:
            tqdm.write(f"Epoch ({i+1}/{args.epochs}): "+
                f"train loss {avg_loss:.5f} / test loss {test_losses[-1]:.5f}")

    torch.save(model.cpu(), f"realnvp_{args.dataset}.pt")
    return train_losses, test_losses, total_norms


def eval(model, data_loader):
    is_training = model.training
    model.eval()
    C, H, W = model.in_shape
    with torch.no_grad():
        total_loss, j = 0., 0
        for x, _ in data_loader:
            log_prob = model.log_prob(x)
            loss = -torch.mean(log_prob) / (C * H * W)
            total_loss += loss.item() * x.shape[0]
            j += x.shape[0]
    if is_training: model.train()
    return total_loss / j


def plot(figname, train_losses, test_losses):
    # Plot the loss during training.
    n_epochs = len(test_losses) - 1
    xs_train = np.linspace(0, n_epochs, len(train_losses))
    xs_test = np.arange(n_epochs+1)

    fig, ax = plt.subplots()
    ax.set_title("Loss value during training")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.plot(xs_train, train_losses, lw=0.6, label="train loss")
    ax.plot(xs_test, test_losses, lw=3., label="test loss")
    ax.legend(loc="upper right")
    fig.savefig(figname)
    plt.close(fig)


def plot_samples(dataset_name):
    model_path = f"realnvp_{dataset_name}.pt"
    fig_path = f"generated_images_{dataset_name}.png"

    model = torch.load(model_path, map_location=torch.device("cuda"))
    imgs = model.sample(n=36)     # img,shape = (36, 3, 32, 32)
    grid = torchvision.utils.make_grid(imgs, nrow=6)

    fig, ax = plt.subplots(figsize=(4.8, 4.8), tight_layout={"pad":0})
    ax.axis("off")
    ax.imshow(grid.permute(1, 2, 0))
    fig.savefig(fig_path)


def plot_total_norm(figname, total_norms):
    n_epochs = len(test_losses) - 1
    xs_train = np.linspace(0, n_epochs, len(train_losses))

    fig, ax = plt.subplots()
    ax.set_title("Total grad norm during training")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Grad norm")
    ax.plot(xs_train, total_norms, lw=0.6)
    fig.savefig(figname)
    plt.close(fig)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=None, type=int)
    parser.add_argument("--lr", default=3e-4, type=float)
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--clip_grad", default=None, type=float)
    parser.add_argument("--verbose", action="store_true", default=False)

    parser.add_argument("--dataset", default="CIFAR10", type=str)
    args = parser.parse_args()

    train_losses, test_losses, total_norms = train(args)
    plot(f"loss_{args.dataset}.png", train_losses, test_losses)
    plot_total_norm(f"total_norm_{args.dataset}.png", total_norms)
    plot_samples(args.dataset)

#