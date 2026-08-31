import random
import numpy as np
import socket, pickle
import torch
from reversi import reversi
from torch import nn

# -----------------------------
# Model 
# -----------------------------
class ReversiPlayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(2, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU()
        )
        self.flatten = nn.Flatten()
        self.shared = nn.Sequential(
            nn.Linear(64*8*8, 128),
            nn.ReLU()
        )
        self.policy = nn.Linear(128, 64)
        self.value = nn.Sequential(
            nn.Linear(128, 1),
            nn.Tanh()

        )

    def forward(self, x):
        x = self.conv(x)
        x = self.flatten(x)
        x = self.shared(x)
        return self.policy(x), self.value(x)

# -----------------------------
# Board → tensor
# -----------------------------
def board_to_tensor(board, turn):
    player = (board == turn).astype(np.float32)
    opponent = (board == -turn).astype(np.float32)
    return np.stack([player, opponent])

# -----------------------------
# Heuristic weights (VERY IMPORTANT)
# -----------------------------
POSITION_WEIGHTS = np.array([
    [100, -20, 10, 5, 5, 10, -20, 100],
    [-20, -50, -2, -2, -2, -2, -50, -20],
    [10,  -2,  5, 1, 1,  5,  -2, 10],
    [5,   -2,  1, 0, 0,  1,  -2, 5],
    [5,   -2,  1, 0, 0,  1,  -2, 5],
    [10,  -2,  5, 1, 1,  5,  -2, 10],
    [-20, -50, -2, -2, -2, -2, -50, -20],
    [100, -20, 10, 5, 5, 10, -20, 100],
])

# -----------------------------
# Main
# -----------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(False)

    game_socket = socket.socket()
    game_socket.connect(('127.0.0.1', 33333))

    game = reversi()
    model = ReversiPlayer().to(device)
    model.load_state_dict(torch.load("reversi_model.pth", map_location=device))
    model.eval()

    while True:
        data = game_socket.recv(4096)
        turn, board = pickle.loads(data)

        if turn == 0:
            game_socket.close()
            return

        game.board = board

        valid_moves = []
        print(turn) 
        print(board)
        for i in range(8):
            for j in range(8):
                if game.step(i, j, turn, False) > 0:
                    valid_moves.append((i, j))

        if not valid_moves:
            x, y = -1, -1

        else:
            best_score = -1e9
            best_move = valid_moves[0]

            states = []
            moves = []

            # -----------------------------
            # Evaluate each move
            # -----------------------------
            for (i, j) in valid_moves:
                original = game.board.copy()

                flips = game.step(i, j, turn, True)

                # Heuristic score
                positional = POSITION_WEIGHTS[i][j]

                # Opponent mobility (IMPORTANT)
                opponent_moves = 0
                for x2 in range(8):
                    for y2 in range(8):
                        if game.step(x2, y2, -turn, False) > 0:
                            opponent_moves += 1

                mobility_score = -5 * opponent_moves

                states.append(board_to_tensor(game.board, -turn))
                moves.append((i, j, flips, positional, mobility_score))

                game.board = original

            # -----------------------------
            # Batch NN evaluation
            # -----------------------------
            states = torch.from_numpy(np.array(states)).to(device)
            _, values = model(states)
            values = values.view(-1)

            # -----------------------------
            # Combine everything
            # -----------------------------
            for idx, (i, j, flips, positional, mobility) in enumerate(moves):
                score = (
                    0.5 * flips +          # greedy baseline
                    2.0 * positional +     # board strategy
                    mobility +             # restrict opponent
                    5.0 * values[idx].item()     # neural evaluation
                )

                if score > best_score:
                    best_score = score
                    best_move = (i, j)

            x, y = best_move

        game_socket.send(pickle.dumps([x, y]))

if __name__ == '__main__':
    main()