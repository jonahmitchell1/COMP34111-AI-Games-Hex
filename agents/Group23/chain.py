from itertools import chain
from src.Board import Board
from src.Colour import Colour

from agents.Group23.utilities import Utilities

class Chain:
    """
    https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=4406406
    Represents a disjoint set of same-coloured connected tiles.
    """
    def __init__(self, board_size: int, colour: Colour):
        self.size = board_size
        self.colour = colour

        self.tiles = set()
        self.is_n_edge = False
        self.is_e_edge = False
        self.is_s_edge = False
        self.is_w_edge = False

        self._influence_region = None

    def __eq__(self, other):
        return isinstance(other, Chain) and self.tiles == other.tiles

    def __hash__(self):
        return hash(tuple(self.tiles))

    def add_tile(self, position: tuple[int, int]):
        if position[0] == 0 :
            self.is_n_edge = True
        if position[1] == 0:
            self.is_w_edge = True
        if position[0] == self.size - 1:
            self.is_s_edge = True
        if position[1] == self.size - 1:
            self.is_e_edge = True

        self.tiles.add(position)

        # Clear the influence region (cache invalidation)
        self.influence_region = None
    
    def add_tiles(self, positions: set[tuple[int, int]]):
        self.tiles |= positions

        # Clear the influence region (cache invalidation)
        self.influence_region = None

    
    def merge_chains(self, chain):
        self.tiles |= chain.tiles
        self.is_n_edge = self.is_n_edge or chain.is_n_edge
        self.is_e_edge = self.is_e_edge or chain.is_e_edge
        self.is_s_edge = self.is_s_edge or chain.is_s_edge
        self.is_w_edge = self.is_w_edge or chain.is_w_edge

        # Clear the influence region (cache invalidation)
        self.influence_region = None

    @property
    def chain_type(self) -> int:
        if self.colour == Colour.RED:
            if self.is_n_edge and self.is_s_edge:
                return 'TopBottom'
            if self.is_n_edge:
                return 'Top'
            if self.is_s_edge:
                return 'Bottom'
        if self.colour == Colour.BLUE:
            if self.is_w_edge and self.is_e_edge:
                return 'LeftRight'
            if self.is_w_edge:
                return 'Left'
            if self.is_e_edge:
                return 'Right'
        return 'Misc'
    
    def get_influence_region(self, board) -> set[tuple[int, int]]:
        if self._influence_region is not None:
            # return cached value
            return self._influence_region

        self._influence_region = set()

        for (x, y) in self.tiles:
            tile = board.tiles[x][y]
            for neighbour in Utilities.get_neighbours(board, tile):
                if neighbour.colour == None:
                    self._influence_region.add((neighbour.x, neighbour.y))

        return self._influence_region