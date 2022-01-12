import numpy as np
from collections import namedtuple


class Cube:
    """
    A concrete class representation of a Rubik's Cube.
    The state of the cube is represented as a flattened 2D matrix.

    0 0 0 1 1 1 2 2 2 3 3 3 4 4 4 5 5 5     # 0 is orange
    0 0 0 1 1 1 2 2 2 3 3 3 4 4 4 5 5 5     # 1 is green  
    0 0 0 1 1 1 2 2 2 3 3 3 4 4 4 5 5 5     # 2 is red
                                            # 3 is white
                                            # 4 is blue
                                            # 5 is yellow
    
    Every color number is additionally represented as a one-hot
    vector, thus making the state a 3D tensor of shape (3, 18, 6).
    """
    #----------- class variables ----------#
    Side = namedtuple("Side", ["LEFT", "FRONT", "RIGHT", "BACK", "UP", "DOWN"])(
                                0, 1, 2, 3, 4, 5)
    Direction = namedtuple("Direction", ["ANTI_CLOCK", "CLOCK"])(0, 1)
    Color = namedtuple("Color", ["ORANGE", "GREEN", "RED", "WHITE", "BLUE", "YELLOW"])(
                                0, 1, 2, 3, 4, 5)

    num_actions = len(Side) * len(Direction)
    action_space = np.arange(num_actions, dtype=int)
    terminal_state = np.hstack([np.full((3,3), _col) for _col in Color])
    terminal_state = np.array(np.expand_dims(
                                    terminal_state, -1) == np.arange(len(Color)),
                                    dtype=np.float32)

    #------------ class methods -----------#
    @classmethod
    def is_terminal(cls, state):
        """ Return True if the state is the terminal state for the environment.
        
        @param state (Array): A state of the environment.
        @returns result (bool): True if the state is terminal for the environment.
                                Otherwise False.
        """
        return np.all(state == cls.terminal_state)

    @classmethod
    def reward(cls, state):
        """ Return the immediate reward on transition to state `state`.

        @param state (Array): A state of the environment.
        @returns reward (int): Reward on trasition to state `state`.
        """
        return 1 if cls.is_terminal(state) else -1

    @classmethod
    def expand_state(cls, state):
        """ Given a state use the model of the environment to obtain
        the descendants of that state and their respective rewards.
        Return the descendants and the rewards.

        @param state (Array): A state of the environment.
        @returns children (Array[state]): A numpy array of shape (num_acts, state.shape) giving
                                            the children of the input state.
        @returns rewards (Array): A numpy array of shape (num_acts, 1) containing the
                                    respective rewards.
        """
        _cube = cls()
        _cube.state = state
        children = [_cube._take_action[act]() for act in cls.action_space]
        rewards = [cls.reward(_c) for _c in children]
        return np.stack(children), np.vstack(rewards)   # rewards shape is (num_acts, 1)

    #----------- env initializer ----------#
    def __init__(self):
        """ Initialize an environment object. """
        # Enumerate all valid actions from the action space.
        self._take_action = {0: self._left_anticlock,
                             1: self._left_clock,
                             2: self._front_anticlock,
                             3: self._front_clock,
                             4: self._right_anticlock,
                             5: self._right_clock,
                             6: self._back_anticlock,
                             7: self._back_clock,
                             8: self._up_anticlock,
                             9: self._up_clock,
                             10: self._down_anticlock,
                             11: self._down_clock}

        # Set the current state to None.
        self._state = None

        # Reset the environment.
        self.reset()

    #---------- property methods ----------#
    @property
    def state(self):
        """ Return the current state of the environment. """
        return self._state.copy()

    @state.setter
    def state(self, new_state):
        """ Set the current state of the environment. """
        self._state = new_state.copy()

    #----------- public methods -----------#
    def step(self, act):
        """ Make a single step taking the specified action.

        @param act (int): An integer value in the range [0, 12).
        @returns next_state (state): The next observed state after taking action `act`.
        @returns reward (int): An integer representing the reward after arriving at the next state.
        @returns done (bool): A boolen indicating whether this is a terminal state.
        """
        if act not in self.action_space:
            raise Exception("Unknown action %s", act)

        # Observe the next state after taking action `act`.
        next_state = self._take_action[act]()

        # Change the current state.
        self._state = next_state.copy()

        # Check if this is the final state.
        done = self.is_solved()
        reward = self.reward(self._state)

        return next_state, reward, done

    def set_random_state(self, scrambles=None):
        """ Set the current state of the environment to a random valid state. """
        self.reset()
        scrambles = 100 if scrambles is None else scrambles
        acts = np.random.randint(low=0, high=self.num_actions, size=scrambles)
        for a in acts:
            self.step(a)

    def reset(self):
        """ Set the current state of the environment to the terminal state. """
        self._state = self.terminal_state.copy()

    def is_solved(self):
        """ Return True if the current state is the terminal state for the environment. """
        return self.is_terminal(self._state)

    #----------- private methods ----------#
    def _left_anticlock(self):
        """ Perform anti-clockwise rotation of the left side.
        Return the resulting state.
        """
        # Unpack cube sides and rotation directions.
        l, f, r, b, u, d = self.Side
        a_cl, cl = self.Direction
        _ = (l, f, r, b, u, d, a_cl, cl)
        next_state = self._state.copy()

        # Rotate the side in the given direction
        side, dir = l, a_cl
        next_state[:, 3*side:3*(side+1)] = np.rot90(self._state[:, 3*side:3*(side+1)], k=(-1)**dir)

        # Rotate the adjecent rows and columns.
        next_state[:, 3 * u] = self._state[:, 3 * f]
        next_state[:, 3 * f] = self._state[:, 3 * d]
        next_state[:, 3 * d] = self._state[:, 3 * b + 2][::-1]      # flip
        next_state[:, 3 * b + 2] = self._state[:, 3 * u][::-1]      # flip

        return next_state

    def _left_clock(self):
        """ Perform clockwise rotation of the left side.
        Return the resulting state.
        """
        # Unpack cube sides and rotation directions.
        l, f, r, b, u, d = self.Side
        a_cl, cl = self.Direction
        _ = (l, f, r, b, u, d, a_cl, cl)
        next_state = self._state.copy()

        # Rotate the side in the given direction
        side, dir = l, cl
        next_state[:, 3*side:3*(side+1)] = np.rot90(self._state[:, 3*side:3*(side+1)], k=(-1)**dir)

        # Rotate the adjecent rows and columns.
        next_state[:, 3 * f] = self._state[:, 3 * u]
        next_state[:, 3 * d] = self._state[:, 3 * f]
        next_state[:, 3 * u] = self._state[:, 3 * b + 2][::-1]      # flip
        next_state[:, 3 * b + 2] = self._state[:, 3 * d][::-1]      # flip

        return next_state

    def _front_anticlock(self):
        """ Perform anti-clockwise rotation of the front side.
        Return the resulting state.
        """
        # Unpack cube sides and rotation directions.
        l, f, r, b, u, d = self.Side
        a_cl, cl = self.Direction
        _ = (l, f, r, b, u, d, a_cl, cl)
        next_state = self._state.copy()

        # Rotate the side in the given direction
        side, dir = f, a_cl
        next_state[:, 3*side:3*(side+1)] = np.rot90(self._state[:, 3*side:3*(side+1)], k=(-1)**dir)

        # Rotate the adjecent rows and columns.
        next_state[:, 3 * l + 2] = self._state[-1, 3 * u : 3 * (u + 1)][::-1]   # flip
        next_state[0, 3 * d : 3 * (d + 1)] = self._state[:, 3 * l + 2]
        next_state[:, 3 * r] = self._state[0, 3 * d : 3 * (d + 1)][::-1]        # flip
        next_state[-1, 3 * u : 3 * (u + 1)] = self._state[:, 3 * r]

        return next_state

    def _front_clock(self):
        """ Perform clockwise rotation of the front side.
        Return the resulting state.
        """
        # Unpack cube sides and rotation directions.
        l, f, r, b, u, d = self.Side
        a_cl, cl = self.Direction
        _ = (l, f, r, b, u, d, a_cl, cl)
        next_state = self._state.copy()

        # Rotate the side in the given direction
        side, dir = f, cl
        next_state[:, 3*side:3*(side+1)] = np.rot90(self._state[:, 3*side:3*(side+1)], k=(-1)**dir)

        # Rotate the adjecent rows and columns.
        next_state[:, 3 * l + 2] = self._state[0, 3 * d : 3 * (d + 1)]
        next_state[0, 3 * d : 3 * (d + 1)] = self._state[:, 3 * r][::-1]        # flip
        next_state[:, 3 * r] = self._state[-1, 3 * u : 3 * (u + 1)]
        next_state[-1, 3 * u : 3 * (u + 1)] = self._state[:, 3 * l + 2][::-1]   # flip

        return next_state

    def _right_anticlock(self):
        """ Perform anti-clockwise rotation of the right side.
        Return the resulting state.
        """
        # Unpack cube sides and rotation directions.
        l, f, r, b, u, d = self.Side
        a_cl, cl = self.Direction
        _ = (l, f, r, b, u, d, a_cl, cl)
        next_state = self._state.copy()

        # Rotate the side in the given direction
        side, dir = r, a_cl
        next_state[:, 3*side:3*(side+1)] = np.rot90(self._state[:, 3*side:3*(side+1)], k=(-1)**dir)

        # Rotate the adjecent rows and columns.
        next_state[:, 3 * f + 2] = self._state[:, 3 * u + 2]
        next_state[:, 3 * d + 2] = self._state[:, 3 * f + 2]
        next_state[:, 3 * b] = self._state[:, 3 * d + 2][::-1]      # flip
        next_state[:, 3 * u + 2] = self._state[:, 3 * b][::-1]      # flip

        return next_state

    def _right_clock(self):
        """ Perform clockwise rotation of the right side.
        Return the resulting state.
        """
        # Unpack cube sides and rotation directions.
        l, f, r, b, u, d = self.Side
        a_cl, cl = self.Direction
        _ = (l, f, r, b, u, d, a_cl, cl)
        next_state = self._state.copy()

        # Rotate the side in the given direction
        side, dir = r, cl
        next_state[:, 3*side:3*(side+1)] = np.rot90(self._state[:, 3*side:3*(side+1)], k=(-1)**dir)

        # Rotate the adjecent rows and columns.
        next_state[:, 3 * f + 2] = self._state[:, 3 * d + 2]
        next_state[:, 3 * d + 2] = self._state[:, 3 * b][::-1]      # flip
        next_state[:, 3 * b] = self._state[:, 3 * u + 2][::-1]      # flip
        next_state[:, 3 * u + 2] = self._state[:, 3 * f + 2]

        return next_state

    def _back_anticlock(self):
        """ Perform anti-clockwise rotation of the back side.
        Return the resulting state.
        """
        # Unpack cube sides and rotation directions.
        l, f, r, b, u, d = self.Side
        a_cl, cl = self.Direction
        _ = (l, f, r, b, u, d, a_cl, cl)
        next_state = self._state.copy()

        # Rotate the side in the given direction
        side, dir = b, a_cl
        next_state[:, 3*side:3*(side+1)] = np.rot90(self._state[:, 3*side:3*(side+1)], k=(-1)**dir)

        # Rotate the adjecent rows and columns.
        next_state[:, 3 * r + 2] = self._state[0, 3 * u : 3 * (u + 1)]
        next_state[-1, 3 * d : 3 * (d + 1)] = self._state[:, 3 * r + 2][::-1]       # flip
        next_state[:, 3 * l] = self._state[-1, 3 * d : 3 * (d + 1)]
        next_state[0, 3 * u : 3 * (u + 1)] = self._state[:, 3 * l][::-1]        # flip

        return next_state

    def _back_clock(self):
        """ Perform clockwise rotation of the back side.
        Return the resulting state.
        """
        # Unpack cube sides and rotation directions.
        l, f, r, b, u, d = self.Side
        a_cl, cl = self.Direction
        _ = (l, f, r, b, u, d, a_cl, cl)
        next_state = self._state.copy()

        # Rotate the side in the given direction
        side, dir = b, cl
        next_state[:, 3*side:3*(side+1)] = np.rot90(self._state[:, 3*side:3*(side+1)], k=(-1)**dir)

        # Rotate the adjecent rows and columns.
        next_state[:, 3 * r + 2] = self._state[-1, 3 * d : 3 * (d + 1)][::-1]   # flip
        next_state[-1, 3 * d : 3 * (d + 1)] = self._state[:, 3 * l]
        next_state[:, 3 * l] = self._state[0, 3 * u : 3 * (u + 1)][::-1]        # flip
        next_state[0, 3 * u : 3 * (u + 1)] = (self._state[:, 3 * r + 2])

        return next_state

    def _up_anticlock(self):
        """ Perform anti-clockwise rotation of the up side.
        Return the resulting state.
        """
        # Unpack cube sides and rotation directions.
        l, f, r, b, u, d = self.Side
        a_cl, cl = self.Direction
        _ = (l, f, r, b, u, d, a_cl, cl)
        next_state = self._state.copy()

        # Rotate the side in the given direction
        side, dir = u, a_cl
        next_state[:, 3*side:3*(side+1)] = np.rot90(self._state[:, 3*side:3*(side+1)], k=(-1)**dir)

        # Rotate the adjecent rows.
        next_state[0, 0:12] = np.roll(self._state[0, 0:12], shift=3, axis=0)

        return next_state

    def _up_clock(self):
        """ Perform clockwise rotation of the up side.
        Return the resulting state.
        """
        # Unpack cube sides and rotation directions.
        l, f, r, b, u, d = self.Side
        a_cl, cl = self.Direction
        _ = (l, f, r, b, u, d, a_cl, cl)
        next_state = self._state.copy()

        # Rotate the side in the given direction
        side, dir = u, cl
        next_state[:, 3*side:3*(side+1)] = np.rot90(self._state[:, 3*side:3*(side+1)], k=(-1)**dir)

        # Rotate the adjecent rows.
        next_state[0, 0:12] = np.roll(self._state[0, 0:12], shift=-3, axis=0)

        return next_state

    def _down_anticlock(self):
        """ Perform anti-clockwise rotation of the down side.
        Return the resulting state.
        """
        # Unpack cube sides and rotation directions.
        l, f, r, b, u, d = self.Side
        a_cl, cl = self.Direction
        _ = (l, f, r, b, u, d, a_cl, cl)
        next_state = self._state.copy()

        # Rotate the side in the given direction
        side, dir = d, a_cl
        next_state[:, 3*side:3*(side+1)] = np.rot90(self._state[:, 3*side:3*(side+1)], k=(-1)**dir)

        # Rotate the adjecent rows.
        next_state[-1, 0:12] = np.roll(self._state[-1, 0:12], shift=-3, axis=0)

        return next_state

    def _down_clock(self):
        """ Perform clockwise rotation of the down side.
        Return the resulting state.
        """
        # Unpack cube sides and rotation directions.
        l, f, r, b, u, d = self.Side
        a_cl, cl = self.Direction
        _ = (l, f, r, b, u, d, a_cl, cl)
        next_state = self._state.copy()

        # Rotate the side in the given direction
        side, dir = d, cl
        next_state[:, 3*side:3*(side+1)] = np.rot90(self._state[:, 3*side:3*(side+1)], k=(-1)**dir)

        # Rotate the adjecent rows.
        next_state[-1, 0:12] = np.roll(self._state[-1, 0:12], shift=3, axis=0)

        return next_state

    #----------- static methods -----------#
    @staticmethod
    def plot_state(state):
        """
        Given an environment state, prints the state in pretty form.

              4 4 4                 # 0 is orange
              4 4 4                 # 1 is green
              4 4 4                 # 2 is red
        0 0 0 1 1 1 2 2 2 3 3 3     # 3 is white
        0 0 0 1 1 1 2 2 2 3 3 3     # 4 is blue
        0 0 0 1 1 1 2 2 2 3 3 3     # 5 is yellow
              5 5 5            
              5 5 5            
              5 5 5            
        """
        result = np.zeros((9, 12), dtype=int)
        state = np.argmax(state, axis=-1)
        result[3:6, :] = state[:, :12]
        result[0:3, 3:6] = state[:, 12:15]
        result[6:9, 3:6] = state[:, 15:]
        print(result)

#