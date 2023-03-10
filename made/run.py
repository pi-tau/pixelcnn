import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as data
import torchvision
from torchvision import transforms
from tqdm import tqdm

from made import MADE


def train(args):
    # Use cuda.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set the random seeds.
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True

    # Create a dataloader for the MNIST dataset.
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: (x > 0.5).to(torch.uint8)), # binarize the dataset.
    ])
    train_data = torchvision.datasets.MNIST(
        "datasets", train=True, download=True, transform=transform)
    test_data = torchvision.datasets.MNIST(
        "datasets", train=False, download=True, transform=transform)
    train_loader = data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    test_loader = data.DataLoader(test_data, batch_size=args.batch_size)

    # Initialize the model.
    C, H, W = (1, 28, 28)
    model = MADE(input_shape=(C, H, W), d=2, hidden_sizes=[512, 512, 512], one_hot=False)
    model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Run the training loop.
    train_losses, test_losses = [], []
    test_losses.append(eval(model, test_loader))
    for i in tqdm(range(args.epochs)):
        avg_loss, j = 0., 0

        # Iterate over the training set. Note that we don't need the labels.
        for x, _ in train_loader:
            log_prob = model.log_prob(x)
            loss = -log_prob.mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            optimizer.step()

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

    torch.save(model, "made.pt")
    return train_losses, test_losses


def eval(model, test_loader):
    is_training = model.training
    model.eval()
    with torch.no_grad():
        total_loss, i = 0., 0
        for x, _ in test_loader:
            log_prob = model.log_prob(x)
            loss = -log_prob.mean()
            total_loss += loss.item()
            i += 1
    if is_training: model.train()
    return total_loss / i


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


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=None, type=int)
    parser.add_argument("--lr", default=3e-4, type=float)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--clip_grad", default=10.0, type=float)
    parser.add_argument("--verbose", action="store_true", default=False)
    args = parser.parse_args()

    train_losses, test_losses = train(args)
    plot("loss.png", train_losses, test_losses)

#