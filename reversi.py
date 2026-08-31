#Zijie Zhang, Sep.24/2023

import numpy as np

class reversi:
    # Initialize the board with:
    # - 8 x 8 grid
    # - places four pieces at center of the board
    #     - 2 black
    #     - 2 white
    def __init__(self) -> None:

        # creates 8 x 8 grid filled with 0s
        self.board = np.zeros([8,8])

        # standard start positions
        self.board[3,4] = -1
        self.board[3,3] = 1
        self.board[4,3] = -1
        self.board[4,4] = 1

        # tracks number of pieces each player has
        #  used for:
        #    - reversi_server.py to dispaly text for number of pieces (respectively)
        self.white_count = 2
        self.black_count = 2

        # directions you can look from a placed piece on the board
        self.directions = [
            [1,1],
            [1,0],
            [1,-1],
            [0,1],
            [0,-1],
            [-1,1],
            [-1,0],
            [-1,-1]
        ]

        # time counter used to display timer 
        self.time = 0

        # whose turn it is (white default)
        #  used for:
        #    - reversi_server.py main loop will flip after a valid move
        self.turn = 1

    # note: step does not automatically flip the turn, rely on reversi_server.py for this
    #
    # x, y: where to place the piece
    # piece: which player is moving (1 - white, -1 - black)
    # commit:
    #    - true - actually modify the board
    #    - false - simulate the move
    #
    # output: 
    #    -1 - space is occupied
    #    -2 - Out of bounds
    #    -3 - no pieces would be flipped (invalid move)
    #    Positive Number - number of pieces that would be flipped 
    #
    # used for:
    #    - greedy_player.py calls this for every (i, j)
    #        - uses return value to decide the best move (most opponent pieces flipped)
    #    - reversi_server.py calls this to actually play the move
    def step(self, x, y, piece = 1, commit = True) -> int:

        #Piece already exists
        if self.board[x,y] != 0:
            return -1
        
        #Out of bound
        elif x < 0 or x > 7 or y < 0 or y > 7:
            return -2
        
        else:

            #opponent pieces flipped
            fliped = 0

            # loop for all 8 directions to check
            for direction in self.directions:
                dx, dy = direction

                # start with 1 step away
                cursor_x, cursor_y = x + dx, y + dy
                # stores coordinates of opponent pieces to be flipped in this direction
                flip_list = []

                # while the space is still in bounds
                while 0 <= cursor_x <=7 and 0 <= cursor_y <=7:

                    # space is empty, no more player/opponent pieces in this direction
                    if self.board[cursor_x, cursor_y] == 0:
                        break

                    # if piece is same color
                    elif self.board[cursor_x, cursor_y] == piece:
                        # if flip_list is empty, there is no sandwich
                        if len(flip_list) == 0:
                            break

                        # if flip_list not empty, go through each coordinate in the flip_list
                        else:
                            for cord in flip_list:
                                # if commit is true, flip the piece to player color
                                if commit:
                                    self.board[*cord] = piece
                                # incrememnt flipped regardless of commit value
                                #    - this needs to be returned for action or determination of action
                                fliped += 1
                            break
                    # otherwise piece is opponent's piece
                    #    - add coordinates to flip_list
                    #    - move the cursor further in the same direction (while loop checks for boundary limit)
                    else:
                        flip_list.append([cursor_x, cursor_y])
                        cursor_x, cursor_y = cursor_x + dx, cursor_y + dy

            #Illegal Move
            if fliped == 0:
                return -3

            # if the move is legal and commit is true - 
            #    - legal is checked by previous if statement
            #    - commit is true only when called from 'reversi_server.py'
            else:
                if commit:
                    # place new piece on the board
                    self.board[x,y] = piece

                    # increase count for current player
                    if piece == 1:
                        self.white_count += 1
                    else:
                        self.black_count += 1

                    # Adjust counts for flipped pieces
                    self.white_count += fliped * piece
                    self.black_count -= fliped * piece
                return fliped