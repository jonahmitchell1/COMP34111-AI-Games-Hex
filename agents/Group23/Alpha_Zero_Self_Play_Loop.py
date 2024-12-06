import logging
from typing import override

from src.Colour import Colour
from agents.Group23.AlphaZeroAgent import AlphaZeroAgent
from src.Colour import Colour
from src.Game import Game
from src.Player import Player
from agents.Group23.Alpha_Zero_NN import Alpha_Zero_NN
from multiprocessing import Pool
import random

logger = logging.getLogger(__name__)
logging.basicConfig(filename='self_play.log', encoding='utf-8', level=logging.info)

class alpha_zero_self_play_loop:

    _board_size: int = 11
    _Student_Network = None
    _Teacher_Network = None
    _max_games_per_simulation = 7
    _simulation_iterations = 50
    _MCTS_turn_length_s = 3
    _game_log_location = "alpha_zero_self_play.log"

    def __init__(self):

        self._Student_Network = Alpha_Zero_NN(board_size=self._board_size)
        self._Teacher_Network = Alpha_Zero_NN(board_size=self._board_size)
        logger.info("Initialised Alpha Zero Self Play Loop")

    def set_up_game(self):
        g = Game(
            player1=Player(
                name="student player",
                agent=AlphaZeroAgent(Colour.RED ,turn_length_s=self._MCTS_turn_length_s , custom_trained_network = self._Student_Network),
            ),
            player2=Player(
                name="teacher player",
                agent=AlphaZeroAgent(Colour.BLUE, turn_length_s=self._MCTS_turn_length_s, custom_trained_network = self._Teacher_Network),
            ),
            board_size=self._board_size,
            logDest=self._game_log_location,
            verbose=True,
            silent=True
        )

        return g

    def _simulate_game(self):
        # simulate game between two agents
        current_game = self.set_up_game()
        current_game.run()

        random_hash = random.getrandbits(128)

        winner_colour = current_game.board.get_winner()

        self._Student_Network._commit_experience_from_buffer(winner_colour=winner_colour, override_path=f"/thread_data/data_{random_hash}.txt")
        self._Teacher_Network._commit_experience_from_buffer(winner_colour=winner_colour, override_path=f"/thread_data/data_{random_hash}.txt")

        return winner_colour


    def _run(self):
        for sim_iter in range(self._simulation_iterations):
            logger.info(f"Simulation iteration {sim_iter+1} of {self._simulation_iterations}")

            results = []

            with Pool(processes=4) as pool:
                # logger.info(f"Game {i+1} of {self._max_games_per_simulation}")
                args = [(self,)] * self._max_games_per_simulation
                results = pool.map(self._simulate_game, args)

            Alpha_Zero_NN.merge_thread_experience_files_in_folder('/thread_data') # merge all thread data into one file
            
            win_count = sum(1 for winner in results if winner == Colour.RED)

            # check majority win rate and swap networks if necessary after 70% win rate
            if win_count / self._max_games_per_simulation > 0.55:
                self._swap_student_teacher_networks()
            else:
                print("Majority win rate not reached, continuing training student without swapping networks")

            # train student network
            self._Student_Network._train()

    def _swap_student_teacher_networks(self):
        # swap student and teacher networks
        logger.info("Swapping student and teacher networks")
        temp = self._Student_Network
        self._Student_Network = self._Teacher_Network
        self._Teacher_Network = temp