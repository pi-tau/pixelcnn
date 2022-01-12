import time
import json
import numpy as np
import jax
import jax.numpy as jnp
from jax.experimental import optimizers
from jax.tree_util import tree_flatten, tree_unflatten, tree_map
from functools import partial
from math import ceil


# from fcnn import fc_net as model_fn
from cnn import conv_net as model_fn



#-------------------- data generation utilities --------------------#
def expand_states(env, states):
    """ Given an array of states use the model of the environment to
    obtain the descendants of these states and their respective rewards.
    Return the descendants and the rewards.

    @param env (Cube Object): A Cube object representing the environment.
    @param states (Array[state]): A numpy array of valid states of the environment.
                                  The shape of the array is (N, state.shape),
                                  where N is the number of states.
    @returns children (Array[state]): A numpy array giving the children of the input states
                                      The shape of the array is (N * num_acts, state.shape).
    @returns rewards (Array): A numpy array of shape (N, num_acts, 1) containing the
                              respective rewards.
    """
    zipped = [env.expand_state(s) for s in states]
    children, rewards = list(zip(*zipped))
    children = np.vstack(children)
    rewards = np.stack(rewards)
    return children, rewards


def generate_episodes(env, episodes, k, trust_range=0):
    """ Generate a random sequence of states starting from the solved state.

    @param env (Cube Object): A Cube object representing the environment.
    @param episodes (int): Number of episodes to be created.
    @param k (int): Length of backward moves.
    @param trust_range (int):
    @returns states (Array[state]): Sequence of generated states. The shape of the array
                                    is (episodes * k, state.shape).
    @returns weights (Array): Array of weights. w[i] corresponds to the weight of states[i].
    @returns children (Array[state]): Sequence of states corresponding to the children of
                                      each of the generated states. The shape of the array
                                      is (episodes * k * num_acts, state.shape).
    @returns rewards (Array): Array of rewards. rewards[i] corresponds to the immediate
                              reward on transition to state children[i]
    """
    # Create an environtment.
    cube = env()
    num_actions = cube.num_actions
    states, w = [], []
    # Create `episodes` number of episodes.
    for _ in range(episodes):
        cube.reset()
        actions = np.random.randint(low=0, high=num_actions, size=k)
        states.extend((cube.step(act)[0] for act in actions))
        # w.extend((1 / (d+1) for d in range(k)))
        w.extend((1.0 if d < trust_range else 1.0 / (d - trust_range + 2) for d in range(k)))
    # Expand each state to obtain children and rewards.
    children, rewards = expand_states(env, states)
    return np.array(states), np.array(w), np.array(children), np.array(rewards)


def make_targets(children, rewards, params):
    """ Generate target values.

    @param children (Array[state]): An array giving the children of the input states
                                    The shape of the array is (N * num_acts, state.shape).
    @param rewards (Array): A numpy array of shape (N, num_acts, 1) containing the
                            respective rewards.
    @param params (pytree): Model parameters for the prediction function.
    @returns vals (Array): An array giving the predicted values of each state.
    """
    # Run the states through the network in batches.
    batch_size = 1024
    vals = []
    for i in range(ceil(children.shape[0] / batch_size)):
        v = apply_fun(params, children[i * batch_size : (i + 1) * batch_size])
        vals.append(v)
    # Add rewards to make target values.
    vals = np.vstack(vals).reshape(rewards.shape) + rewards
    return np.max(vals, axis=1)


def batch_generator(data, batch_size):
    """ Yields random batches of data.

    @param data (Dict): Input dataset. Dictionary with keys "X", "y", "w".
    @param batch_size (int): Size of batches to be generated.
    @yields batch (Dict): Random batch of data of size `batch_size`.
    """
    num_train = data["X"].shape[0]
    while True:
        idxs = np.random.choice(np.arange(num_train), size=batch_size, replace=False)
        yield (data["X"][idxs],
               data["y"][idxs],
               data["w"][idxs])



#------------------------ utility functions ------------------------#
def compute_trust_range(env, params, num_cubes=1000):
    scrambles = 1
    cube = env()
    while True:
        for _ in range(num_cubes):
            # Start at a random state with scrambles.
            cube.set_random_state(scrambles)
            solved = False
            # Try to solve the cubes using at most `scrambles` moves
            for _ in range(scrambles):
                children, rewards = env.expand_state(cube._state)
                vals = apply_fun(params, children)
                act = int(np.argmax(vals + rewards))
                cube.step(act)
                if cube.is_solved():
                    solved = True
                    break
            if not solved:
                return scrambles - 1
        # If all the cubes were solved then increment `s`.
        scrambles += 1


# def fib(n, memo={}):
#     """ Return the n-th Fibonacci number. """
#     if n == 0 or n == 1:
#         memo[n] = 1
#     elif n not in memo:
#         memo[n] = fib(n-1, memo) + fib(n-2, memo)
#     return memo[n]


# def reverse_fib(n):
#     """ Return the index of the greatest number from the Fibonacci sequence,
#     that is smaller than or equal to n. """
#     i = 0
#     while fib(i+1) <= n:
#         i += 1
#     return i



#-------------------- optimizer and LR schedule --------------------#
step_size = 1e-2
decay_rate = 0.65       # 0.65 ** 10 = 0.01 ---> decaying the step size 10 times ammounts to dividing by 100
decay_steps = 10
step_fn = optimizers.exponential_decay(step_size=step_size,
                                       decay_rate=decay_rate,
                                       decay_steps=decay_steps)
opt_init, opt_update, get_params = optimizers.nesterov(step_size=step_fn, mass=0.9)



#-------------------- params training utilities --------------------#
reg = 3e-5
clip_max_grad = 10.0


init_fun, apply_fun = model_fn()
apply_fun = jax.jit(apply_fun)


@jax.jit
def l2_regularizer(params, reg=reg):
    """ Return the L2 regularization loss. """
    leaves, _ = tree_flatten(params)
    return reg * jnp.sum(jnp.array([jnp.vdot(x, x) for x in leaves]))


@jax.jit
def loss_fn(params, batch):
    """ Return the total loss computed for a given batch. """
    X, y, w = batch
    vals = apply_fun(params, X)
    mse_loss = jnp.mean(((vals - y) ** 2).squeeze() * w)
    l2_loss = l2_regularizer(params)
    return mse_loss + l2_loss


@jax.jit
def update(i, opt_state, batch):
    """ Perform backpropagation and parameter update. """
    params = get_params(opt_state)
    loss, grads = jax.value_and_grad(loss_fn)(params, batch)
    # clipped_grads = optimizers.clip_grads(grads, clip_norm)   # norm clipping produces very big differences when jit.
    clipped_grads = tree_map(lambda w: jnp.clip(w, -clip_max_grad, clip_max_grad), grads)
    return loss, opt_update(i, clipped_grads, opt_state)


def train(env, batch_size=128, num_epochs=5, num_iterations=21,
          num_samples=101, print_every=10, episodes=1000, k_max=25,
          verbose=False, params_save_path=""):
    """
    Train the model function by generating simulations of random-play.
    On every epoch generate a new simulation and run multiple iterations.
    On every iteration evaluate the targets using the most recent model parameters
    and run multiple times through the dataset.
    At the end of every epoch check the performance and store the best performing params.
    If the performance drops then decay the step size parameter.

    @param env (Cube Object): A Cube object representing the environment.
    @param batch_size (int): Size of minibatches used to compute loss and gradient during training.         [optional]
    @param num_epochs (int): The number of epochs to run for during training.                               [optional]
    @param num_iterations (int): The number of iterations through the generated episodes.                   [optional]
    @param num_samples (int): The number of times the dataset is reused.                                    [optional]
    @param print_every (int): An integer. Training progress will be printed every `print_every` iterations. [optional]
    @param episodes (int): Number of episodes to be created.                                                [optional]
    @param k_max (int): Maximum length of sequence of backward moves.                                       [optional]
    @param clip_norm (float): A scalar for gradient clipping.                                               [optional]
    @param verbose (bool): If set to false then no output will be printed during training.                  [optional]
    @param params_save_path (str): File path to save the model parameters.                                  [optional]
    @returns params (pytree): The best model parameters obatained during training.
    @returns loss_history (List): Loss history of iter_mean_loss and fisrt_loss computed during training.
    """
    loss_history = {"iter_loss" : [], "first_loss" : []}
    trust_range = 1

    # Initialize model parameters.
    params = list(jnp.load(params_save_path + "params_cnn_0.npy", allow_pickle=True)) \
                if params_save_path is not None \
                else init_fun(jax.random.PRNGKey(0), env.terminal_state.shape)[1]

    # Begin training.
    for e in range(num_epochs):
        tic = time.time()

        # Initialize the optimizer state at the begining of each epoch.
        opt_state = opt_init(params)

        # Generate data from random-play using the environment.
        states, w, children, rewards = generate_episodes(env, episodes, k_max, trust_range)

        # Train the model on the generated data. Periodically recompute the target values.
        epoch_mean_loss = 0.0
        for it in range(num_iterations):
            tic_it = time.time()

            # Make targets for the generated episodes using the most recent params and build a batch generator.
            params = get_params(opt_state)
            tgt_vals = make_targets(children, rewards, params)
            data = {"X" : states, "y" : tgt_vals, "w" : w}
            train_batches = batch_generator(data, batch_size)

            # Run through the dataset and update model params.
            total_loss = 0.0
            for i in range(num_samples):
                batch = next(train_batches)
                loss, opt_state = update(e, opt_state, batch)
                total_loss += loss
                if it == 0 and i == 0:
                    loss_history["first_loss"].append(loss)
                    print("First loss: {:.3f}".format(loss))

            # Book-keeping.
            iter_mean_loss = total_loss / num_samples
            epoch_mean_loss = (it * epoch_mean_loss + iter_mean_loss) / (it + 1)
            loss_history["iter_loss"].append(iter_mean_loss)

            # Printout results.
            toc_it = time.time()
            if print_every != 0 and it % print_every == 0 and verbose:
                print("\t(Iteration({}/{}) took {:.3f} seconds) iter_mean_loss = {:.3f}".format(
                                                        it + 1, num_iterations, (toc_it-tic_it), iter_mean_loss))

        # Recompute the trust range using latest model params.
        trust_range = compute_trust_range(env, params)

        # Store the model parameters.
        if params_save_path is not None:
            jnp.save(params_save_path + "params_cnn_%d" % (e + 1), params)

        # Record the time needed for a single epoch.
        toc = time.time()

        # Printout results.
        if verbose:
            print("(Epoch ({}/{}) took {:.3f} seconds), epoch_mean_loss: {:.3f}, trust_range: {}".format(
                                                        e + 1, num_epochs, (toc-tic), epoch_mean_loss, trust_range))

    # Save loss history.
    json.dump(loss_history, open(params_save_path + "loss_history.json", "w"), indent=2)

    return params, loss_history

#