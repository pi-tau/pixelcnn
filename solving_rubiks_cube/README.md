# SOLVING RUBIK'S CUBE USING REINFORCEMENT LEARNING

This repository contains python implementation of a model using reinforcement learning to learn to solve Rubik's cube.
A convolutional neural network is used as a value function approximator. The neural network is trained on data generated from self-play and used as a heuristic to guide an A* search algorithm.

The code is heavily inspired from the following papers:
 * [1] **Solving the rubik's cube with approximate policy iteration** by S. McAleer, et. al.
 * [2] **Solving the rubik's cube with depp reinforcement learning and search** by F. Agostinelli, et. al.

The following modules are implemented:
 * `cube_model_naive.py` implements the cube model environment
 * `cnn.py` implements the value function approximator using jax
 * `Astar.py` implements A* solver for the rubik's cube
 * `train.py` encapsulates all the logic needed to train the solver