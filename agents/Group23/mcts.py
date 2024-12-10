from copy import deepcopy
import random
import time
import os

from src.Board import Board
from src.Colour import Colour
from src.Move import Move

from multiprocessing.pool import Pool

from agents.Group23.treenode import TreeNode
import numpy as np

class MCTS:
    """Implements the Monte Carlo Tree Search algorithm."""

    def __init__(self, colour: Colour, max_simulation_length: float = 2.5, custom_trained_network=None):
        self.colour = colour  # Agent's colour
        self.max_simulation_length = max_simulation_length  # Length of a MCTS search in seconds


    def _process_simulations(self, root, start_time):
        """Processes the results of the simulations."""

        processed_simulations = 0

        while time.time() - start_time < self.max_simulation_length:
            node = self._select(root)
            result = self._simulate(node)
            self._backpropagate(node, result)
            processed_simulations += 1

        return (root, processed_simulations)

    def run(self, board: Board):
        root = TreeNode(board=board, player=self.colour)

        iterations = 0
        start_time = time.time()

        # Run simulations in parallel
        with Pool(os.cpu_count()) as pool:
            results = pool.starmap(self._process_simulations, [(root, start_time)] * os.cpu_count())

            children = {}
            
            for result in results:
                processed_root, processed_simulations = result
                iterations += processed_simulations
                
                for child in processed_root.children:
                    if child.move not in children:
                        children[child.move] = child
                    else:
                        children[child.move].visits += child.visits
                        children[child.move].wins += child.wins

            root.children = list(children.values())
            
        print(f'Ran {iterations} simulations in {time.time() - start_time:.2f}s')

        best_child = max(
            root.children,
            key=lambda child: (
                (np.sum(child.wins) / np.sum(child.visits)) if np.sum(child.visits) > 0 else float('-inf')
            )
        )

        print(f'Selected move with {best_child.visits} visits and {best_child.wins} wins from {len(root.children)} possible moves')
        print(f'Moves:')
        # for child in root.children:
        #     print(f'  - Move: ({child.move.x, child.move.y}), Wins: {child.wins}, Visits: {child.visits}')

        return best_child.move, None

    def _select(self, node: TreeNode):
        """Selects a node to expand using the UCT formula."""
        moves = self.get_heuristic_moves(node)
        while node.is_fully_expanded(moves):
            node = node.best_child()
        return self._expand(node)

    def _expand(self, node: TreeNode):
        """Expands the node by adding a new child."""
        moves = self.get_heuristic_moves(node)
        unvisited_moves = [move for move in moves if move not in [child.move for child in node.children]]

        if len(unvisited_moves) > 0:
            new_move = random.choice(unvisited_moves)
            return node.add_child(new_move)

        return node

    def _simulate(self, node: TreeNode):
        """Simulates a random game from the current node and returns the result."""
        simulation_board = deepcopy(node.board)

        # Play randomly until the game ends
        current_colour = self.colour.opposite()
        while (not simulation_board.has_ended(colour=current_colour) and
               not simulation_board.has_ended(colour=current_colour.opposite())):
            moves = self.get_all_moves(simulation_board)

            move = self._default_policy(moves)

            x, y = move.x, move.y
            simulation_board.set_tile_colour(x, y, current_colour)
            current_colour = current_colour.opposite()

        return 1 if simulation_board.get_winner() == self.colour else 0

    def _backpropagate(self, node: TreeNode, result: int):
        """Backpropagates the simulation result through the tree."""
        while node is not None:
            node.visits += 1
            node.wins += result
            node = node.parent
            result = 1 - result  # Invert the result for the opponent's perspective

    def get_all_moves(self, board: Board) -> list[Move]:
        choices = [
            (i, j) for i in range(board.size) for j in range(board.size)
        ]
        return [Move(x, y) for x, y in choices if board.tiles[x][y].colour == None]
    
    def get_heuristic_moves(self, node: TreeNode) -> list[Move]:
        """
        Generates a subset of all legal moves for the current board state, based on the heuristic given:
        https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=4406406
        """
        moves = node.moves

        if len(moves) == 0:
            moves = self.get_all_moves(node.board)

        return moves

    def _default_policy(self, moves: list[Move]) -> Move:
        """
        Implements a default policy to select a simulation move.
        """
        if len(moves) == 0:
            raise ValueError("No legal moves available")
        return random.choice(moves)