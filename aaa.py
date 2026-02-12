import tkinter as tk
from tkinter import messagebox

BOARD_SIZE = 8
CELL_SIZE = 60
EMPTY = 0
BLACK = 1
WHITE = 2

DIRECTIONS = [(-1, -1), (-1, 0), (-1, 1),
              (0, -1),          (0, 1),
              (1, -1),  (1, 0), (1, 1)]


class OthelloGame:
    def __init__(self):
        self.board = [[EMPTY] * BOARD_SIZE for _ in range(BOARD_SIZE)]
        self.board[3][3] = WHITE
        self.board[3][4] = BLACK
        self.board[4][3] = BLACK
        self.board[4][4] = WHITE
        self.current_player = BLACK

    def is_on_board(self, r, c):
        return 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE

    def get_flips(self, r, c, player):
        if self.board[r][c] != EMPTY:
            return []
        opponent = WHITE if player == BLACK else BLACK
        all_flips = []
        for dr, dc in DIRECTIONS:
            flips = []
            nr, nc = r + dr, c + dc
            while self.is_on_board(nr, nc) and self.board[nr][nc] == opponent:
                flips.append((nr, nc))
                nr += dr
                nc += dc
            if flips and self.is_on_board(nr, nc) and self.board[nr][nc] == player:
                all_flips.extend(flips)
        return all_flips

    def get_valid_moves(self, player):
        moves = []
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if self.get_flips(r, c, player):
                    moves.append((r, c))
        return moves

    def place(self, r, c, player):
        flips = self.get_flips(r, c, player)
        if not flips:
            return False
        self.board[r][c] = player
        for fr, fc in flips:
            self.board[fr][fc] = player
        return True

    def count(self):
        b = sum(cell == BLACK for row in self.board for cell in row)
        w = sum(cell == WHITE for row in self.board for cell in row)
        return b, w


class OthelloGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("オセロ")
        self.root.resizable(False, False)
        self.game = OthelloGame()

        self.info_label = tk.Label(root, text="", font=("Arial", 16))
        self.info_label.pack(pady=5)

        canvas_size = CELL_SIZE * BOARD_SIZE
        self.canvas = tk.Canvas(root, width=canvas_size, height=canvas_size, bg="green")
        self.canvas.pack(padx=10, pady=(0, 10))
        self.canvas.bind("<Button-1>", self.on_click)

        self.score_label = tk.Label(root, text="", font=("Arial", 14))
        self.score_label.pack(pady=(0, 10))

        self.draw_board()

    def draw_board(self):
        self.canvas.delete("all")
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                x1 = c * CELL_SIZE
                y1 = r * CELL_SIZE
                x2 = x1 + CELL_SIZE
                y2 = y1 + CELL_SIZE
                self.canvas.create_rectangle(x1, y1, x2, y2, outline="black", fill="green")

                piece = self.game.board[r][c]
                if piece != EMPTY:
                    color = "black" if piece == BLACK else "white"
                    pad = 4
                    self.canvas.create_oval(x1 + pad, y1 + pad, x2 - pad, y2 - pad, fill=color)

        # 有効な手をハイライト
        valid_moves = self.game.get_valid_moves(self.game.current_player)
        for r, c in valid_moves:
            cx = c * CELL_SIZE + CELL_SIZE // 2
            cy = r * CELL_SIZE + CELL_SIZE // 2
            dot_r = 5
            self.canvas.create_oval(cx - dot_r, cy - dot_r, cx + dot_r, cy + dot_r,
                                    fill="yellow", outline="yellow")

        player_name = "黒" if self.game.current_player == BLACK else "白"
        self.info_label.config(text=f"手番: {player_name}")

        b, w = self.game.count()
        self.score_label.config(text=f"黒: {b}  |  白: {w}")

    def on_click(self, event):
        c = event.x // CELL_SIZE
        r = event.y // CELL_SIZE
        if not self.game.is_on_board(r, c):
            return

        if not self.game.place(r, c, self.game.current_player):
            return

        self.game.current_player = WHITE if self.game.current_player == BLACK else BLACK

        # 次のプレイヤーが置けるかチェック
        if not self.game.get_valid_moves(self.game.current_player):
            # パス: 相手に戻す
            self.game.current_player = WHITE if self.game.current_player == BLACK else BLACK
            if not self.game.get_valid_moves(self.game.current_player):
                # 両者とも置けない → ゲーム終了
                self.draw_board()
                self.show_result()
                return
            else:
                passed = "黒" if (WHITE if self.game.current_player == BLACK else BLACK) == BLACK else "白"
                messagebox.showinfo("パス", f"{passed} は置ける場所がありません。パスします。")

        self.draw_board()

    def show_result(self):
        b, w = self.game.count()
        if b > w:
            msg = f"黒の勝ち！ ({b} vs {w})"
        elif w > b:
            msg = f"白の勝ち！ ({b} vs {w})"
        else:
            msg = f"引き分け！ ({b} vs {w})"
        self.info_label.config(text="ゲーム終了")
        if messagebox.askyesno("ゲーム終了", f"{msg}\n\nもう一度プレイしますか？"):
            self.game = OthelloGame()
            self.draw_board()


if __name__ == "__main__":
    root = tk.Tk()
    OthelloGUI(root)
    root.mainloop()
