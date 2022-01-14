import time
import sys

import numpy as np
import torch
# import torch_xla.core.xla_model as xm   # tpu

from src.utils import batch_iter, eval_ppl


def train(model, data, learning_rate, lr_decay, clip_grad, batch_size, max_epochs,
          max_num_trial, patience_limit, model_save_path, stdout=sys.stdout):
    """Run optimization to train the model. Save the best performing model parameters.

    Args:
        model (NMT): NMT object.
        data (Dict): A dataset object.
        learning_rate (float): A scalar giving the learning rate.
        lr_decay (float): A scalar for exponentially decaying the learning rate.
        clip_grad (float): A scalar for gradient clipping.
        batch_size (int): Size of minibatches used to compute loss and gradient during training.
        max_epochs (int): The number of epochs to run for during training.
        max_num_trial (int): The number of trials before termination.
        patience_limit (int): The number of epochs to wait before returning to the best model.
        model_save_path (str): File path to save the model.
        stdout (file, optional): File object (stream) used for standard output of logging
            information. Default value is `sys.stdout`.
    """
    # Put the model in training mode.
    model.train()

    # Check if 'cuda' is available.
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # device = xm.xla_device() # tpu
    print("Using device: %s" % device, file=stdout)

    # Send the model to device.
    model = model.to(device)

    # Initialize the optimizer.
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=lr_decay)

    # Begin training.
    print("Begin training..", file=stdout)
    epoch = patience = num_trial = 0
    best_dev_ppl = 0.
    while True:
        tic = time.time()
        epoch += 1

        for src_sents, tgt_sents in batch_iter(data["train_data"], batch_size, shuffle=True):
            report_loss = report_examples = cum_tgt_words = 0.
            curr_batch_size = len(src_sents)

            # Compute the forward pass and the loss.
            total_loss = model(src_sents, tgt_sents)
            loss = total_loss / curr_batch_size

            # Zero the gradients, perform backward pass, clip the gradients, and update the gradients.
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            optimizer.step()

            # Bookkeeping.
            report_loss += total_loss.item()
            report_examples += curr_batch_size
            tgt_word_num_to_predict = sum(len(s[1:]) for s in tgt_sents)    # omitting leading "<s>"
            cum_tgt_words += tgt_word_num_to_predict

        # At the end of every epoch evaluate the model perplexity on the development set.
        avg_loss = report_loss / report_examples
        train_ppl = np.exp(report_loss / cum_tgt_words)
        dev_ppl = eval_ppl(model, data["dev_data"])
        toc = time.time()

        # Printout results.
        print("Epoch (%d/%d), train time: %.2f min, avg loss: %.1f, avg train ppl: %.1f, dev ppl: %.1f" % (
            epoch, max_epochs, (toc-tic)/60, avg_loss, train_ppl, dev_ppl), file=stdout)

        # If the model is performing better than it was on the previous epoch, then save
        # the model parameters and the optimizer state.
        # If the model is performing worse than it was on the previous epoch, then
        # increase the patience. if the patience reaches a limit decay the learning rate
        # and increase trial count.
        if best_dev_ppl == 0 or dev_ppl < best_dev_ppl:
            best_dev_ppl = dev_ppl
            print(f"saving the new best model to [{model_save_path}]..", file=stdout)
            model.save(model_save_path)
            torch.save(optimizer.state_dict(), model_save_path + ".optim")
            patience = 0
        else:
            patience += 1
            print("increasing patience = %d" % patience, file=stdout)
            if patience >= patience_limit:
                # Reset patience.
                patience = 0

                # Increase the trial count.
                num_trial += 1
                print("increasing num trial: %d" % num_trial, file=stdout)

                # Decay the learning rate.
                lr_scheduler.step()

        # If the trial count reaches the maximum number of trials, stop the training.
        if num_trial >= max_num_trial:
            print("Reached maximum number of trials!", file=stdout)
            break

        # If the maximum number of epochs is reached, stop the training.
        if epoch == max_epochs:
            print("Reached maximum number of epochs!", file=stdout)
            break

    # Load the best saved model.
    print("loading the best performing model..", file=stdout)
    params = torch.load(model_save_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(params["state_dict"])
    model.train()
    model = model.to(device)

#