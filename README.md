# vacuum_world_search
Implementation of DFS, A*, and IDA* for the Vacuum World problem
# CYS5550 – Homework 1
# Option A: Vacuum World Search Algorithms
# Complete, self-contained implementation
# Standard library only

from collections import deque
import heapq
from typing import Tuple, Set, List, Optional, FrozenSet

############################################################
# Vacuum World Problem Definition
############################################################

Action = str
Position = Tuple[int, int]
State = Tuple[Position, FrozenSet[Position]]  # (robot_pos, remaining_dirty_cells)


class VacuumWorld:
    def __init__(self, grid, start_pos: Position, dirty_cells: Set[Position]):
        """
        grid: 2D list where 0 = free cell, 1 = obstacle
        start_pos: (row, col)
        dirty_cells: set of (row, col)
        """
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0])
        self.start_pos = start_pos
        self.initial_state: State = (start_pos, frozenset(dirty_cells))

    def in_bounds(self, r, c):
        return 0 <= r < self.rows and 0 <= c < self.cols

    def is_free(self, r, c):
        return self.grid[r][c] == 0

    def is_goal(self, state: State) -> bool:
        _, dirty = state
        return len(dirty) == 0

    def successors(self, state: State):
        """
        Returns list of (action, next_state, cost)
        """
        (r, c), dirty = state
        successors = []

        moves = {
            'UP': (-1, 0),
            'DOWN': (1, 0),
            'LEFT': (0, -1),
            'RIGHT': (0, 1)
        }

        for action, (dr, dc) in moves.items():
            nr, nc = r + dr, c + dc
            if self.in_bounds(nr, nc) and self.is_free(nr, nc):
                successors.append((action, ((nr, nc), dirty), 1))

        # CLEAN action
        if (r, c) in dirty:
            new_dirty = set(dirty)
            new_dirty.remove((r, c))
            successors.append(('CLEAN', ((r, c), frozenset(new_dirty)), 1))

        return successors


############################################################
# Heuristic Function (Admissible)
############################################################

def heuristic_manhattan(state: State) -> int:
    """
    Heuristic = distance to nearest dirty cell + number of remaining dirty cells - 1
    (admissible and simple)
    """
    (r, c), dirty = state
    if not dirty:
        return 0

    distances = [abs(r - dr) + abs(c - dc) for dr, dc in dirty]
    return min(distances) + (len(dirty) - 1)


############################################################
# Depth-First Search (DFS)
############################################################

def dfs_search(problem: VacuumWorld):
    stack = [(problem.initial_state, [], 0)]
    explored = set()

    nodes_expanded = 0
    max_frontier_size = 1

    while stack:
        max_frontier_size = max(max_frontier_size, len(stack))
        state, path, cost = stack.pop()

        if state in explored:
            continue
        explored.add(state)

        nodes_expanded += 1

        if problem.is_goal(state):
            return path, nodes_expanded, max_frontier_size

        for action, next_state, step_cost in problem.successors(state):
            if next_state not in explored:
                stack.append((next_state, path + [action], cost + step_cost))

    return None, nodes_expanded, max_frontier_size


############################################################
# A* Search
############################################################

class PriorityQueue:
    def __init__(self):
        self.heap = []
        self.entry_finder = {}
        self.counter = 0

    def push(self, item, priority):
        if item in self.entry_finder:
            self.remove(item)
        entry = [priority, self.counter, item]
        self.counter += 1
        self.entry_finder[item] = entry
        heapq.heappush(self.heap, entry)

    def remove(self, item):
        entry = self.entry_finder.pop(item)
        entry[-1] = None

    def pop(self):
        while self.heap:
            _, _, item = heapq.heappop(self.heap)
            if item is not None:
                del self.entry_finder[item]
                return item
        raise KeyError("pop from empty priority queue")

    def __len__(self):
        return len(self.entry_finder)


def astar_search(problem: VacuumWorld, heuristic):
    frontier = PriorityQueue()
    frontier.push(problem.initial_state, heuristic(problem.initial_state))

    came_from = {problem.initial_state: None}
    action_from = {}
    g_cost = {problem.initial_state: 0}

    nodes_expanded = 0
    max_frontier_size = 1

    while len(frontier) > 0:
        max_frontier_size = max(max_frontier_size, len(frontier))
        current = frontier.pop()

        nodes_expanded += 1

        if problem.is_goal(current):
            path = []
            while came_from[current] is not None:
                path.append(action_from[current])
                current = came_from[current]
            path.reverse()
            return path, nodes_expanded, max_frontier_size

        for action, next_state, cost in problem.successors(current):
            new_g = g_cost[current] + cost
            if next_state not in g_cost or new_g < g_cost[next_state]:
                g_cost[next_state] = new_g
                f = new_g + heuristic(next_state)
                frontier.push(next_state, f)
                came_from[next_state] = current
                action_from[next_state] = action

    return None, nodes_expanded, max_frontier_size


############################################################
# IDA* Search
############################################################

def idastar_search(problem: VacuumWorld, heuristic):
    def dfs(state, g, threshold, path, visited):
        f = g + heuristic(state)
        if f > threshold:
            return f
        if problem.is_goal(state):
            return path

        minimum = float('inf')
        for action, next_state, cost in problem.successors(state):
            if next_state not in visited:
                visited.add(next_state)
                result = dfs(next_state, g + cost, threshold, path + [action], visited)
                if isinstance(result, list):
                    return result
                minimum = min(minimum, result)
                visited.remove(next_state)
        return minimum

    threshold = heuristic(problem.initial_state)
    iterations = 0
    nodes_expanded = 0

    while True:
        iterations += 1
        visited = {problem.initial_state}
        result = dfs(problem.initial_state, 0, threshold, [], visited)
        if isinstance(result, list):
            return result, nodes_expanded, iterations
        if result == float('inf'):
            return None, nodes_expanded, iterations
        threshold = result


############################################################
# Example Test Case (4x4 Grid from Assignment)
############################################################

if __name__ == '__main__':
    # 0 = free, 1 = obstacle
    grid = [
        [0, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 0]
    ]

    start = (0, 0)
    dirty = {(0, 2), (1, 3), (2, 0), (3, 2)}

    problem = VacuumWorld(grid, start, dirty)

    print("DFS:", dfs_search(problem))
    print("A* :", astar_search(problem, heuristic_manhattan))
    print("IDA*:", idastar_search(problem, heuristic_manhattan))
