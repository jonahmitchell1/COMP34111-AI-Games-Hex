from random import choice

from src.AgentBase import AgentBase
from src.Board import Board
from src.Colour import Colour
from src.Move import Move


class AlphaBetaAgent(AgentBase):
    """This class implements an alpha-beta search agent to play Hex."""

    _choices: list[Move]
    _board_size: int = 11
    _DEPTH: int = 10

    def __init__(self, colour: Colour):
        super().__init__(colour)
        self._choices = [
            (i, j) for i in range(self._board_size) for j in range(self._board_size)
        ]

    def make_move(self, turn: int, board: Board, opp_move: Move | None) -> Move:
        """The game engine will call this method to request a move from the agent.
        If the agent is to make the first move, opp_move will be None.
        If the opponent has made a move, opp_move will contain the opponent's move.
        If the opponent has made a swap move, opp_move will contain a Move object with x=-1 and y=-1,
        the game engine will also change your colour to the opponent colour.

        Args:
            turn (int): The current turn
            board (Board): The current board state
            opp_move (Move | None): The opponent's last move

        Returns:
            Move: The agent's move
        """

        current_player = (turn % 2) + 1
        best_move = None
        best_score = int("-inf") if (current_player == 1) else int("inf")    # player 1 is the maximiser; player 2 is the minimiser
        depth = self._DEPTH

        alpha = int("-inf")
        beta = int("inf")

        # Search through every possible move
        for index, (i, j) in enumerate(self._choices):
            # make the move (i, j)
            valid_moves = self._choices.copy()
            valid_moves.pop(index)
            board.set_tile_colour(i, j, self.colour)

            # perform minimax algorithm
            score = self.miniMaxAlphaBeta(self.turn + 1, Colour.opposite(self.colour), board, valid_moves, depth - 1, alpha, beta)

            # revert move on board
            board.set_tile_colour(None)

            # update best move
            if current_player == 1 and score > best_score:
                best_score = score
                best_move = Move(i, j)

                # update alpha
                alpha = max(score, alpha)
            elif current_player == 2 and score < best_score:
                best_score = score
                best_move = Move(i, j)
                
                # update beta
                beta = min(score, beta)

        
        return best_move
    
    def miniMaxAlphaBeta(self, turn: int, colour: Colour, board: Board, valid_moves, depth, alpha, beta) -> Move:
        """Returns the best score that can be achieved for the current player at a certain serach depth using the minimax algorithm.
        Player 1 is the maximiser; tries to get a greatest score
        Player 2 is the minimiser; tries to get the smallest score
        
        Args:
            turn (int): The current turn
            colour (Colour): The colour of the current player to make their move
            board (Board): The current board state
            valid_moves (list): List of possible moves the current player can make
            depth (int): Search depth of the algorithm

        Returns:
            score (int): The best score that can be achieved from the current position for the current player.
        """

        # Base case: maximum depth reached / no more valid moves left
        if depth == 0 or len(valid_moves) == 0:
            return self.evaluateBoard(board)

        # Step case: depth-limited search through all valid moves and taking the best move out of them all
        current_player = (turn % 2) + 1
        best_score = int("-inf") if (current_player == 1) else int("inf")    # player 1 is the maximiser; player 2 is the minimiser

        # Search through every possible move
        for index, (i, j) in enumerate(valid_moves):
            # make the move (i, j)
            updated_valid_moves = valid_moves.copy()
            updated_valid_moves.pop(index)
            board.set_tile_colour(i, j, colour)

            # continue minimax search down this path
            score = self.miniMaxAlphaBeta(turn + 1, Colour.opposite(colour), board, updated_valid_moves, depth - 1)
            
            # revert move on board
            board.set_tile_colour(None)

            # update best move
            if current_player == 1:
                # maximise score
                best_score = max(score, best_score)
                alpha = max(score, alpha)

                if beta <= alpha:
                    break
            if current_player == 2:
                # minimise score
                best_score = min(score, best_score)
                beta = min(score, beta)

                if beta <= alpha:
                    break
        
        return best_score
    
    def evaluateBoard(self, board):
        # TODO Implement board evaluation heuristic
        pass
