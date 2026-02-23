"""
Tic Tac Toe Player
"""


import math


X = "X"
O = "O"
EMPTY = None




def initial_state():
   """
   Returns starting state of the board.
   """
   return [[EMPTY, EMPTY, EMPTY],
           [EMPTY, EMPTY, EMPTY],
           [EMPTY, EMPTY, EMPTY]]




def player(board):
   """
   Returns player who has the next turn on a board.
   """


   # if the board state is empty then player X is chosen
   if board == initial_state():
       return X


   # If the board is not empty, count how many X's and O's have been played
   x_count = sum(1 for row in board for cell in row if cell == X)
   o_count = sum(1 for row in board for cell in row if cell == O)


   # Which ever player has the fewest moves will be allocated
   return X if x_count <= o_count else O




def actions(board):
   """
   Returns set of all possible actions (i, j) available on the board.
   """
   return {
       (i, j)
       for i in range(len(board))
       for j in range(len(board[i]))
       if board[i][j] is EMPTY
   }




def result(board, action):
   """
   Returns the board that results from making move (i, j) on the board.
   """
   i, j = action


   # 1) Validity check
   if (i, j) not in actions(board):
       raise ValueError(f"Invalid action: cell ({i}, {j}) is not empty")


   # 2) Deep‐copy the board
   # Option A: using copy.deepcopy
   new_board = [row[:] for row in board]


   # 3) Apply the move for the correct player
   new_board[i][j] = player(board)


   return new_board




def winner(board):
   """
   Returns the winner of the game, if there is one.
   """
   for row in board:
       if row[0] is not EMPTY and row[0] == row[1] == row[2]:
           return row[0]


       # Check columns
   for j in range(3):
       if board[0][j] is not EMPTY and board[0][j] == board[1][j] == board[2][j]:
           return board[0][j]


       # Check diagonals
   if board[0][0] is not EMPTY and board[0][0] == board[1][1] == board[2][2]:
       return board[0][0]
   if board[0][2] is not EMPTY and board[0][2] == board[1][1] == board[2][0]:
       return board[0][2]


       # Check for draw (full board, no winner)
   full = all(cell is not EMPTY for row in board for cell in row)
   if full:
       return None


   # No winner and still empty spaces
   return None




def terminal(board):
   """
   Returns True if game is over, False otherwise.
   """
   # If someone has won, it’s over.
   if winner(board) is not None:
       return True


   # If there are no legal moves left, it’s a draw → over.
   if not actions(board):
       return True


   # Otherwise the game should continue.
   return False




def utility(board):
   """
   Returns 1 if X has won the game, -1 if O has won, 0 otherwise.
   """
   win = winner(board)
   if win == X:
       return 1
   elif win == O:
       return -1
   else:
       return 0




def minimax(board):
   """
   Returns the optimal action for the current player on the board.
   """


   if terminal(board):
       return None


   # Helper for the maximizer (X)
   def max_value(state):
       if terminal(state):
           return utility(state), None
       v = -math.inf
       best_action = None
       for action in actions(state):
           min_v, _ = min_value(result(state, action))
           if min_v > v:
               v = min_v
               best_action = action
               # X can’t do better than a guaranteed win
               if v == 1:
                   break
       return v, best_action


   # Helper for the minimizer (O)
   def min_value(state):
       if terminal(state):
           return utility(state), None
       v = math.inf
       best_action = None
       for action in actions(state):
           max_v, _ = max_value(result(state, action))
           if max_v < v:
               v = max_v
               best_action = action
               # O can’t do better than forcing X’s loss
               if v == -1:
                   break
       return v, best_action


   # Dispatch based on whose turn it is
   if player(board) == X:
       _, move = max_value(board)
   else:
       _, move = min_value(board)


   return move