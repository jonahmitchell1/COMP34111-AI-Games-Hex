from random import choice

from src.AgentBase import AgentBase
from src.Board import Board
from src.Colour import Colour
from src.Move import Move


class TDLearningAgent(AgentBase):
    """This class implements an TD learning agent to play Hex."""

    _choices: list[Move]
    _board_size: int = 11
    _learning_rate = 0.1
    _discount_rate = 0.9
    _value_table: dict[str, float]

    def __init__(self, colour: Colour):
        super().__init__(colour)
        self._choices = [
            (i, j) for i in range(self._board_size) for j in range(self._board_size)
        ]
    
    def get_board_state_key(self, board: Board) -> str:
        """Get the key for the value table based on the given board state.

        Args:
            board (Board): The given board state

        Returns:
            str: The key for the value table
        """
        key = str(board.print_board())
        # init value table if unseen state (default value can be any number unless it is a terminal state)
        ## For this skeleton implementation, we will use 0 for all unseen states
        if key not in self._value_table:
            self._value_table[key] = 0
        return key

    def make_move(self, turn: int, board: Board, opp_move: Move | None) -> Move:
        """Make a move chosen by TD learning.
        if swap possible, swap
        if not, make a move based on the value table
        We update values on every time step
        We complete an episode when the simulated game ends

        Args:
            turn (int): The current turn
            board (Board): The current board state
            opp_move (Move | None): The opponent's last move

        Returns:
            Move: The agent move
        """

        if turn == 2:
            return Move(-1, -1) # swap move


        while True:
            x, y = choice(self._choices) # random walk
            
            # init value table if unseen state
            original_state_key = self.get_board_state_key(board)

            # make the move and get the new state
            board.set_tile_colour(x,y, self.colour)
            new_state_key = self.get_board_state_key(board)

            # quantify the reward
            reward = self.get_state_value(board)

            # update the value table
            self._value_table[original_state_key] = self._value_table[original_state_key] + self._learning_rate * (reward + (self._discount_rate * self._value_table[new_state_key]) - self._value_table[original_state_key])

            # check if the game is over
            if board.is_game_over():
                break

        # choose the next move based on the value table
        best_move = None
        best_value = -1
        for move in self._choices:
            board_copy = board.copy()
            board_copy.set_tile_colour(move[0], move[1], self.colour)
            state_key = self.get_board_state_key(board_copy)
            if self._value_table[state_key] > best_value:
                best_value = self._value_table[state_key]
                best_move = Move(move[0],move[1])

        return best_move
    
    def get_state_value(self, board: Board):
        """Get the value of the current state.

        Args:
            board (Board): The current board state

        Returns:
            float: The value of the state
        """

        if board.is_game_over() and board.get_winner() == self.colour:
            return 1
        elif board.is_game_over() and board.get_winner() != self.colour:
            return -1
        return 0.5
