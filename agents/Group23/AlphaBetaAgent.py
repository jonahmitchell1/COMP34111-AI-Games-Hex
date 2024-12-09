from random import choice

from src.AgentBase import AgentBase
from src.Board import Board
from src.Colour import Colour
from src.Move import Move

from collections import deque, defaultdict


# ========= HEURISTIC METHODS =========
class Heuristics:
    NETWORK_FLOW_HEURISTIC = "network_flow_heuristic"
    TWO_DISTANCE_HEURISTIC = "two_distance_heuristic"

# === NETWORK FLOW HEURISTIC ===
# In this implementation, we create a flow network that represents the board state. We find the maximum flow heuristic by using the Edmonds-Karp algorithm
"""
A class to represent the flow network of the board. Nodes are represented by a position (x, y), and edges are represented by a connection ((x', y'), c).
"""
class FlowNetwork():
    # Represents the network by mapping NetworkNodes to NetworkEdges
    network = {}

    def __init__(self, vertices, edges):
        # Add vertices
        for vertex in vertices:
            self.network[vertex] = []
        
        # Add edges
        for edge in edges:
            vertexStem = edge[0]
            self.network[vertexStem].append((edge[1], edge[2]))
    
    def getEdges(self, vertex):
        return self.network[vertex]

    def removeVertex(self, vertex):
        self.network[vertex] = None
    
    def removeEdge(self, edge):
        vertexStem, vertexEnd = edge[0], edge[1]
        edges = self.network[vertexStem]
        isEdgeRemoved = False

        for i, currentEdge in enumerate(edges):
            if currentEdge[0] == vertexStem and currentEdge[1] == vertexEnd:
                edges.pop(i)
                isEdgeRemoved = True
        
        if not isEdgeRemoved:
            print("Error: attempted to remove an edge that does not exist in the network. Cancelling the operation.")
        else:
            self.network[vertexStem] = edges

    def updateEdges(self, edges):
        self.clearEdges()
        for edge in edges:
            vertexStem = edge[0]
            self.network[vertexStem].append((edge[1], edge[2]))
    def clearEdges(self):
        for key in self.network.keys():
            self.network[key] = []

    # Finds maximum flow using the Edmonds-Karp algorithm
    def findMaximumFlow(self):
        source = (0, 0)
        sink = (0, 10)
        # Prepare capacity and flow matrices
        capacity = defaultdict(lambda: defaultdict(int))
        for u in self.network:
            for v, cap in self.network[u]:
                capacity[u][v] += cap  # Handle multi-edges by summing capacities
        
        # Initialize flow and residual graph
        flow = defaultdict(lambda: defaultdict(int))
        
        def bfs():
            """Finds an augmenting path using BFS and returns it along with the minimum capacity."""
            parent = {}  # To store the path
            visited = set()
            queue = deque([((source), float('inf'))])  # (current node, flow so far)
            
            while queue:
                u, flow_so_far = queue.popleft()
                if u in visited:
                    continue
                visited.add(u)
                
                for v in capacity[u]:
                    residual_capacity = capacity[u][v] - flow[u][v]
                    if v not in visited and residual_capacity > 0:  # Can traverse
                        parent[v] = u
                        new_flow = min(flow_so_far, residual_capacity)
                        if v == sink:  # Reached the sink
                            return parent, new_flow
                        queue.append((v, new_flow))
            
            return None, 0
        
        max_flow = 0
        
        # Augment flow until no more augmenting paths
        while True:
            parent, path_flow = bfs()
            if path_flow == 0:
                break  # No more augmenting paths
            
            max_flow += path_flow
            # Update flow and residual graph
            v = sink
            while v != source:
                u = parent[v]
                flow[u][v] += path_flow
                flow[v][u] -= path_flow  # Reverse flow for residual capacity
                v = u
        
        return max_flow


# ========= ALPHA-BETA SERACH AGENT =========
class AlphaBetaAgent(AgentBase):
    """This class implements an alpha-beta search agent to play Hex."""

    _choices: list[Move]
    _board_size: int = 11
    _DEPTH: int = 2
    _heuristic = Heuristics.TWO_DISTANCE_HEURISTIC#.NETWORK_FLOW_HEURISTIC # or Heuristics.TWO_DISTANCE_HEURISTIC
    _network = None   # used for the network flow heuristic

    def initialise_network(self):
        start_vertex = [(0, 0)]
        end_vertex = [(0, 10)]
        other_vertices = [(i, j) for i in range(0, 11) for j in range(1, 10)]
        all_vertices = start_vertex + end_vertex + other_vertices

        # create edges between vertices
        edges = []
        # start and end connections
        for i in range(0, 11):
            # creating edges for the start node
            edges.append(((0, 0), (i, 1), 1))
            edges.append(((i, 1), (0, 0), 1))

            # creating edges for the end node
            edges.append(((0, 10), (i, 9), 1))
            edges.append(((i, 9), (0, 10), 1))
            
        # connections in between start and end nodes nodes
        for i in range(0, 11):
            for j in range(1, 10):
                # left
                if i - 1 >= 0:
                    edges.append(((i, j), (i-1, j), 1))
                # right
                if i + 1 <= 10:
                    edges.append(((i, j), (i+1, j), 1))
                # up
                if j - 1 >= 1:
                    edges.append(((i, j), (j-1), 1))
                # down
                if j + 1 <= 9:
                    edges.append(((i, j), (i, j+1), 1))
                # diagonal-down
                if i - 1 >= 0 and j + 1 <= 9:
                    edges.append(((i, j), (i-1, j+1), 1))
                # diagonal-up
                if i + 1 <= 10 and j - 1 >= 1:
                    edges.append(((i, j), (i+1, j-1), 1))
        
        # create the network
        self._network = FlowNetwork(all_vertices, edges)

    def __init__(self, colour: Colour):
        super().__init__(colour)
        self._choices = [
            (i, j) for i in range(self._board_size) for j in range(self._board_size)
        ]

        if self._heuristic == Heuristics.NETWORK_FLOW_HEURISTIC:
            self.initialise_network()

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

        # update valid moves
        if opp_move != None:
            for i, move in enumerate(self._choices):
                if move[0] == opp_move.x and move[1] == opp_move.y:
                    self._choices.pop(0)
                    break

        # remove already taken tiles from choices list
        valid_moves = []
        for row in board._tiles:
            for i in range(len(row)):
                if row[i]._colour == None:
                    valid_moves.append( (row[i]._x, row[i]._y) )
        self._choices = valid_moves

        current_player = (turn % 2) + 1
        best_move = None
        best_move_index = 0
        best_score = float("-inf") if (current_player == 1) else float("inf")    # player 1 is the maximiser; player 2 is the minimiser
        depth = self._DEPTH

        alpha = float("-inf")
        beta = float("inf")

        # Search through every possible move
        for index, (i, j) in enumerate(self._choices):
            # make the move (i, j)
            valid_moves = self._choices.copy()
            valid_moves.pop(index)
            board.set_tile_colour(i, j, self.colour)

            # perform minimax algorithm
            score = self.miniMaxAlphaBeta(turn + 1, Colour.opposite(self.colour), board, valid_moves, depth - 1, alpha, beta)

            # revert move on board
            board.set_tile_colour(i, j, None)

            # update best move
            if current_player == 1 and score > best_score:
                best_score = score
                best_move = Move(i, j)
                best_move_index = index

                # update alpha
                alpha = max(score, alpha)
            elif current_player == 2 and score < best_score:
                best_score = score
                best_move = Move(i, j)
                best_move_index = index
                
                # update beta
                beta = min(score, beta)
        
        # Perform the move
        self._choices.pop(best_move_index)

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
        best_score = float("-inf") if (current_player == 1) else float("inf")    # player 1 is the maximiser; player 2 is the minimiser

        # Search through every possible move
        for index, (i, j) in enumerate(valid_moves):
            # make the move (i, j)
            updated_valid_moves = valid_moves.copy()
            updated_valid_moves.pop(index)
            board.set_tile_colour(i, j, colour)

            # continue minimax search down this path
            score = self.miniMaxAlphaBeta(turn + 1, Colour.opposite(colour), board, updated_valid_moves, depth - 1, alpha, beta)
            
            # revert move on board
            board.set_tile_colour(i, j, None)

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
    
    def create_edge_capacity(self, player_colour, tile_colour_stem, tile_colour_end):
        # both tiles are taken by the player
        if tile_colour_stem == player_colour and tile_colour_end == player_colour:
            return 1000
        # a tile is untaken
        elif tile_colour_stem == None or tile_colour_end == None:
            return 1
        
        # either tile is the opponent's tile - no edge
        return 0

    def create_network(self, board, colour):
        """
        Creates the graph network to represent the current board state in the perspective the player.
        :param board: represents the current board state
        :param colour: represents the player's colour
        """
        # create edges between vertices
        edges = []
        # start and end connections
        for i in range(0, 11):
            # creating edges for the start node
            capacity = self.create_edge_capacity(colour, colour, board.tiles[i][1].colour)
            
            edges.append(((0, 0), (i, 1), capacity))
            edges.append(((i, 1), (0, 0), capacity))

            # creating edges for the end node
            capacity = self.create_edge_capacity(colour, board.tiles[i][9].colour, colour)

            edges.append(((0, 10), (i, 9), capacity))
            edges.append(((i, 9), (0, 10), capacity))
            
        # connections in between start and end nodes nodes
        for i in range(0, 11):
            for j in range(1, 10):
                # left
                if i - 1 >= 0:
                    capacity = self.create_edge_capacity(colour, board.tiles[i][j].colour, board.tiles[i-1][j].colour)
                    edges.append(((i, j), (i-1, j), capacity))
                # right
                if i + 1 <= 10:
                    capacity = self.create_edge_capacity(colour, board.tiles[i][j].colour, board.tiles[i+1][j].colour)
                    edges.append(((i, j), (i+1, j), capacity))
                # up
                if j - 1 >= 1:
                    capacity = self.create_edge_capacity(colour, board.tiles[i][j].colour, board.tiles[i][j-1].colour)
                    edges.append(((i, j), (j-1), capacity))
                # down
                if j + 1 <= 9:
                    capacity = self.create_edge_capacity(colour, board.tiles[i][j].colour, board.tiles[i][j+1].colour)
                    edges.append(((i, j), (i, j+1), capacity))
                # diagonal-down
                if i - 1 >= 0 and j + 1 <= 9:
                    capacity = self.create_edge_capacity(colour, board.tiles[i][j].colour, board.tiles[i-1][j+1].colour)
                    edges.append(((i, j), (i-1, j+1), capacity))
                # diagonal-up
                if i + 1 <= 10 and j - 1 >= 1:
                    capacity = self.create_edge_capacity(colour, board.tiles[i][j].colour, board.tiles[i+1][j-1].colour)
                    edges.append(((i, j), (i+1, j-1), capacity))

        # Update network edges  
        self._network.updateEdges(edges)
        return self._network
    
    def evaluateBoard(self, board):
        if self._heuristic == Heuristics.NETWORK_FLOW_HEURISTIC:
            # Initialise the network
            network = self.create_network(board, self.colour)

            # Find the maximum flow heuristic
            heuristic = network.findMaximumFlow()

            return heuristic
        
        if self._heuristic ==Heuristics.TWO_DISTANCE_HEURISTIC:

            return self.TwoDistance(board)


    def TwoDistance(self, board):
        # Using paper https://www.cs.cornell.edu/~adith/docs/y_hex.pdf QueenBee "two-distance" method

        # convert to node format    
        # graph with key (x,y) contains all the neighbour tiles
        #====================================================================================================
        tiles2DArray = board._tiles   #[x][y]
        graph = {}
        for i in range(len(tiles2DArray)): #columns
            for j in range(len(tiles2DArray[i])): #rows
                neighbours = []
                for x,y in zip(tiles2DArray[i][j].I_DISPLACEMENTS,tiles2DArray[i][j].J_DISPLACEMENTS):
                    #dont add out of bounds neighbours
                    if x+i < 0 or x+i > board._size-1 or y+j < 0 or y+j > board._size-1:
                        continue   
                    neighbours.append(tiles2DArray[i+x][j+y])
                graph[(i,j)] = neighbours
        

        # join/cut relevant edges of graph depending on colour of tile
        #=====================================================================================================

        RedGraph = self.cutGraph(graph, Colour.RED, tiles2DArray)
        RedGraph = self.reduceGraph(RedGraph, Colour.RED, tiles2DArray)
        BlueGraph = self.cutGraph(graph, Colour.BLUE, tiles2DArray)
        BlueGraph = self.reduceGraph(BlueGraph, Colour.BLUE, tiles2DArray)

        # using reduced graph, find second shortest path
        #======================================================================================================

        redDistance = self.getSecondDistance(RedGraph, Colour.RED, tiles2DArray)
        blueDistance = self.getSecondDistance(BlueGraph, Colour.BLUE, tiles2DArray)

        if self.colour == Colour.RED:
            return blueDistance - redDistance
        else:
            return redDistance-blueDistance
    
    def getSecondDistance(self, graph, colour, tiles2DArray):
        #add the two nodes to get graph distance between.
        if colour == Colour.RED:
            graph["Start"] = [(0,0),(1,0),(2,0),(3,0),(4,0),(5,0),(6,0),(7,0),(8,0),(9,0),(10,0)]
            graph["End"] = [(0,10),(1,10),(2,10),(3,10),(4,10),(5,10),(6,10),(7,10),(8,10),(9,10),(10,10)]
        if colour == Colour.BLUE:
            graph["Start"] = [(0,0), (0,1), (0,2), (0,3), (0,4), (0,5), (0,6), (0,7), (0,8), (0,9), (0,10)]
            graph["End"] = [(10,0), (10,1), (10,2), (10,3), (10,4), (10,5), (10,6), (10,7), (10,8), (10,9), (10,10)]
        

        shortestDistance = 100000
        shortestPath = None
        loopCount = 0 #prevent infinite loop if there is only 1 path

        #bfs
        from collections import deque
        queue = deque([("Start", ["Start"])])
        visited = set()

        while queue:
            current_node, path = queue.popleft()

            # skip visited nodes
            if current_node in visited:
                continue
            visited.add(current_node)

            if current_node in graph["End"]:
                #instead of ending immediatly find the first path with length +1 of the shortest
                if shortestPath == None:
                    shortestPath = path
                    shortestDistance = len(path)
                    continue
                if len(path) > shortestDistance:
                    path.append("End")
                    break #path is the path taken
                if loopCount >= 15:
                    path = shortestPath
                    path.append("End")
                    break
                else:
                    loopCount += 1


            #add unvisited neighbours to queue
            for neighbor in graph.get(current_node, []):
                if neighbor not in visited:
                    queue.append((neighbor, path + [neighbor]))

        #print(graph["End"])
        #print(path)
        #print(len(path))
        return len(path)
        #while True:
        #    continue


    def reduceGraph(self, graph, colour, tiles2DArray):
        
        #this method is jank im sorry

        graphJoined = {key: [] for key in graph}

        for coords, neighbours in graph.items():
            tile = tiles2DArray[coords[0]][coords[1]]
            
            if tile._colour == None:
                for neighbour in neighbours:
                    if neighbour._colour == None:
                        graphJoined[coords].append((neighbour._x,neighbour._y)) #dont change unclaimed tile 
            elif tile._colour == colour:
               for i, neighbour1 in enumerate(neighbours):
                 for neighbour2 in neighbours[i + 1:]: 
                    if neighbour2 not in graphJoined[(neighbour1._x,neighbour1._y)] and neighbour2._colour == None:
                        graphJoined[(neighbour1._x,neighbour1._y)].append((neighbour2._x,neighbour2._y))#(neighbour2._x,neighbour2._y))
                    if neighbour1 not in graphJoined[(neighbour2._x,neighbour2._y)] and neighbour1._colour == None:
                        graphJoined[(neighbour2._x,neighbour2._y)].append((neighbour1._x,neighbour1._y))
            
            # print(f"Graph: {graphJoined}")
            # print(f"Tile: {tile}")
            # print(f"Neighbours: {neighbours}")
            # print(f"Joined Graph: {graphJoined}")
            # print("")
        return graphJoined
    
    def cutGraph(self, graph, colour, tiles2DArray):
        
        # for a tile, remve any neighbours of opposing colour
        graphCut = {}
        for coords, neighbours in graph.items():
            tile = tiles2DArray[coords[0]][coords[1]] #get tile object
            if tile._colour == None:
                graphCut[coords] = neighbours
                continue #dont change unclaimed nodes
            
            neighbours = [neighbour for neighbour in neighbours if tile._colour == neighbour._colour or neighbour._colour == None]
            graphCut[coords] = neighbours
        return graphCut

