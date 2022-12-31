import torch
import torch.nn.functional as F
import torch.utils.data as data
from tqdm import tqdm


def train(model, train_data, test_data, train_args, verbose=False):
    epochs = train_args["epochs"]
    lr = train_args["lr"]
    batch_size = train_args["batch_size"]
    grad_clip = train_args["grad_clip"]

    train_loader = data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = data.DataLoader(test_data, batch_size=batch_size)

    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_losses, test_losses = [], []
    test_losses.append(eval(model, test_loader))
    for i in tqdm(range(epochs)):
        # Iterate over the training set.
        avg_loss, j = 0., 0
        for x in train_loader:
            x = x.to(model.device).contiguous()
            logits = model(x)
            loss = F.cross_entropy(logits, x.long(), reduction="mean")

            optimizer.zero_grad()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

            avg_loss += loss.item()
            j += 1
        avg_loss /= j

        # Test on the test set.
        test_losses.append(eval(model, test_loader))

        # Maybe printout results.
        if verbose:
            tqdm.write(f"Epoch {i+1}/{epochs}:")
            tqdm.write(f"  train loss {avg_loss} / test loss {test_losses[-1]}")

    return train_losses, test_losses

def eval(model, test_loader):
    is_training = model.training
    model.eval()
    with torch.no_grad():
        total_loss, i = 0., 0
        for x in test_loader:
            x = x.to(model.device).contiguous()
            logits = model(x)
            loss = F.cross_entropy(logits, x.long())
            total_loss += loss.item()
            i += 1
    if is_training: model.train()
    return total_loss / i

#