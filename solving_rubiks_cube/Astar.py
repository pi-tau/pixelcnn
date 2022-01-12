"""Implementation of A* heuristic search algorithm."""
import numpy as np
import heapq
import jax
import jax.numpy as jnp
from collections import abc, deque, defaultdict
from functools import partial, total_ordering
from time import time

class Frontier:
    """
    Data structure supporting POPMAX and UPDATE operations in O(log(N)) time.
    """
    #                           Set             Heap        Frontier
    # Pop best node             O(n)            O(log)      O(log)
    # Update by node            O(1)            O(n)        O(log)
    #
    def __init__(self, nodes={}):
        """
        Initializes a A* search frontier from (node, score) dictionary.
        """
        heap = []
        reverse_index = {}
        i = 0
        for node, value in nodes.items():
            heap.append((value, node))
            reverse_index[node] = i
            i += 1
        self._heap = heap
        self._reverse_index = reverse_index
        self._heapify()
        # self._verify()

    def __len__(self):
        return len(self._heap)

    def __setitem__(self, node, score):
        """
        Updates the value of `node` with `score`, maintaining the heap
        and reverse map invariants.
        """
        try:
            i = self._reverse_index[node]
        except KeyError:
            self._heap.append((score, node))
            j = len(self._heap) - 1
            self._reverse_index[node] = j
            self._bubble(j)
            # self._verify()
        else:
            oldval = self._heap[i][0]
            self._heap[i] = (score, self._heap[i][1])
            if score > oldval:
                self._bubble(i, verify=False)
            elif score < oldval:
                self._sink(i, verify=False)
            # self._verify()

    def __repr__(self):
        return 'Frontier({})'.format(repr(self._heap))

    def pop_best(self):
        """ Remove and return the current best node. """
        H, R = self._heap, self._reverse_index
        # Elements in H are (score, node) tuples
        _, best_node = H[0]
        result = best_node.copy()
        if len(H) == 1:
            R.clear()
            H.clear()
        else:
            R.pop(H[0][1])
            H[0] = H[-1]
            R[H[0][1]] = 0
            H.pop()
        self._sink(0, verify=False)
        return result

    def _sink(self, i, verify=False):
        """
        Moves the item at index `i` downward, maintaining the
        heap invariant and the index map
        """
        # Elements in H are (score, node) tuples
        H = self._heap
        N = len(H)
        while i < N:
            li, ri = (2 * (i + 1) - 1), (2 * (i + 1))
            max_score = H[i][0]
            j = i
            if li < N and H[li][0] > max_score:
                j = li
                max_score = H[li][0]
            if ri < N and H[ri][0] > max_score:
                j = ri
            if j != i:
                index = self._reverse_index
                # Update the index map
                index[H[j][1]], index[H[i][1]] = i, j
                # Do the swap
                H[j], H[i] = H[i], H[j]
                # Next iteration
                i = j
            else:
                break
        if verify:
            self._verify()

    def _bubble(self, i, verify=False):
        """
        Moves the item at index `i` upward, maintaining the
        heap invariant and the index map
        """
        # Elements in H are (score, node) tuples
        H = self._heap
        while i > 0:
            p = (i - 1) // 2
            if H[p][0] < H[i][0]:
                # Update the reverse index
                index = self._reverse_index
                index[H[p][1]], index[H[i][1]] = i, p
                # Do the swap
                H[i], H[p] = H[p], H[i]
                # Next iteration
                i = p
            else:
                break
        if verify:
            self._verify()

    def _heapify(self):
        """ Builds a max heap in `self._heap` """
        H = self._heap
        j = (len(H) - 2) // 2
        sink = self._sink
        for i in range(j, -1, -1):
            sink(i)
        # self._verify()

    def _verify(self):
        """ Verifies the heap and reverse index invariants """
        H = self._heap
        N = len(H)
        R = self._reverse_index
        assert(N == len(R))
        for i in range(N):
            assert(R[H[i][1]] == i)
            li, ri = (2 * (i + 1) - 1), (2 * (i + 1))
            if li < N:
                assert(H[i] >= H[li])
            if ri < N:
                assert(H[i] >= H[ri])


class AStarNode:

    def __init__(self, cube, score=np.inf):
        self._env = type(cube)
        self._cube = type(cube)()
        self._cube.state = cube.state.copy()
        self.score = score

    def __hash__(self):
        return hash(self._cube.state.tostring())

    def __eq__(self, other):
        return np.all(self._cube.state == other._cube.state)

    def copy(self):
        return AStarNode(self._cube, self.score)

    def neighbours(self):
        """ Returns the neighbour nodes of `self`. """
        children, _ = self._cube.expand_state(self._cube.state)
        env = self._env
        result = []
        for child in children:
            x = env()
            x.state = child
            result.append(AStarNode(x))
        return result

    def neighbour_states(self):
        """ Returns the neighbour states of `self._cube.state`. """
        children, _ = self._cube.expand_state(self._cube.state)
        return children


def search(start, heuristic, max_iterations=10_000):
    """
    Description
    -----------
    Returns a path from `start` to solved cube if one exists else [].
    The path contains the actions that leed to the solved state.
    Parameters
    ----------
    start :
        The start state. Must be instance of one of the Rubick Cube's environments.
    heuristic :
        Heuristic function accepting list of states (Numpy array) as input
        and returning the goodness of them as Numpy array.
        Use `heuristic_from_nn()` to create this kind of function from
        JAX `apply()` and `params`.
    max_iterations :
        If no solved state is encountered in less than `max_itertions`
        the search terminates and () is returned.
    Returns
    -------
        Tuple of actions. Actions are integers.
    """
    start_score = float(heuristic([start.state]))
    start_node = AStarNode(start, score=start_score)
    parents = {start_node: (None, None)}

    # distance from `start` to current node
    G = defaultdict(lambda: np.inf)
    G[start_node] = 0
    # frontier = {start_node: start_score}
    # getter = frontier.__getitem__
    frontier = Frontier({start_node: start_score})

    for _ in range(max_iterations):
        # current_node = max(frontier, key=getter)
        # frontier.pop(current_node)
        current_node = frontier.pop_best()
        if current_node._cube.is_solved():
            return _reconstruct_path(current_node, parents)
        neighbour_nodes  = current_node.neighbours()
        neighbour_states = jnp.array(current_node.neighbour_states())
        # estimated distance to solved state. Larger is better
        neighbour_values = heuristic(neighbour_states)
        tentative_score = G[current_node] + 1
        for a in range(12):
            node = neighbour_nodes[a]
            if tentative_score < G[node]:
                parents[node] = (current_node, a)
                G[node] = tentative_score
                node.score = float(neighbour_values[a]) - tentative_score
                frontier[node] = node.score
    return ()


def heuristic_from_nn(apply, params):
    """ Returns heauristic function for `search` from JAX `apply()` and `params`. """
    def H(states):
        return apply(params, jnp.array(states)).ravel()
    return H


def _reconstruct_path(node, parents):
    path = []
    current = node
    while True:
        parent, parent_action = parents[current]
        current = parent
        if parent is None and parent_action is None:
            break
        path.append(parent_action)
    return tuple(path[::-1])