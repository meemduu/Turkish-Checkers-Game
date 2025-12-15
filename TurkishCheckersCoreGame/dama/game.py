import pygame
from .constants import WHITE, BLACK, GREY, SQUARE_SIZE, ROWS, COLS
from dama.board import Board

class Game:
    def __init__(self, win):
        self._init()
        self.win = win
    
    def update(self):
        self.board.draw(self.win)
        self.draw_valid_moves(self.valid_moves)
        pygame.display.update()
    
    def _init(self):
        self.selected = None
        self.board = Board()
        self.turn = WHITE
        self.valid_moves = {}
    
    def winner(self):
        return self.board.winner()
    
    def reset(self):
        self._init()

    def get_board(self):
        return self.board
    """
    def select(self, row, col):
        if self.selected:
            result = self._move(row, col)
            if not result:
                self.selected = None
                self.select(row, col)
        
        piece = self.board.get_piece(row, col)
        if piece != 0 and piece.color == self.turn:
            self.selected = piece
            self.valid_moves = self.board.get_valid_moves(piece)
            return True
            
        return False
           
            
    def _move(self, row, col):
        piece = self.board.get_piece(row, col)
        if self.selected and piece == 0 and (row, col) in self.valid_moves:
            self.board.move(self.selected, row, col)
            skipped = self.valid_moves[(row, col)]
            if skipped:
                self.board.remove(skipped)
            self.change_turn()    
        else:
            return False
        
        return True
    """
    
    def draw_valid_moves(self, moves):
        for move in moves:
            row, col = move
            pygame.draw.circle(self.win, GREY, (col * SQUARE_SIZE + SQUARE_SIZE//2, row * SQUARE_SIZE + SQUARE_SIZE//2), 15)
        
    def change_turn(self):
        self.valid_moves = {}
        if self.turn == WHITE:
            self.turn = BLACK
        else:
            self.turn = WHITE
    
    
    
    
    
    def get_all_valid_moves(self, player_color):
        """
        Belirtilen oyuncu için tüm taşların geçerli hamlelerini hesaplar.
        Dönen yapı:
          { piece1: { move1: [yakalanan taşlar listesi], move2: [...] },
            piece2: { ... },
            ... }
        """
        moves = {}
        for piece in self.get_all_pieces(player_color):
            piece_moves = self.board.get_valid_moves(piece)
            if piece_moves:
                moves[piece] = piece_moves
        return moves
     
    def filter_max_capture_moves(self, moves_dict):
        """
        moves_dict:
          { piece1: { move1: [yakalanan taşlar listesi], move2: [...] },
            piece2: { ... },
            ... }
        
        Eğer herhangi bir hamlede yakalama varsa, yalnızca en fazla
        yakalama sağlayan hamleleri döndürür.
        """
        max_capture = 0

        # Öncelikle tüm hamleler arasında maksimum yakalama sayısını belirleyelim
        for piece_moves in moves_dict.values():
            for captured in piece_moves.values():
                if len(captured) > max_capture:
                    max_capture = len(captured)

        # Eğer hiçbir hamlede taş yakalanmıyorsa, doğrudan orijinal moves_dict döneriz.
        if max_capture == 0:
            return moves_dict

        # Şimdi her taş için, sadece max_capture kadar taş yeme imkanı sağlayan hamleleri saklayalım.
        filtered_moves = {}
        for piece, piece_moves in moves_dict.items():
            # Filtreleme yapıyoruz: sadece len(captured) == max_capture olan hamleler kalmalı.
            valid_moves = {move: captured for move, captured in piece_moves.items() if len(captured) == max_capture}
            if valid_moves:
                filtered_moves[piece] = valid_moves

        return filtered_moves
    
    def select(self, row, col):
        # Oyuncunun tüm taşlarının hamlelerini hesapla ve filtrele
        all_moves = self.get_all_valid_moves(self.turn)
        all_moves = self.filter_max_capture_moves(all_moves)

        # Eğer daha önceden bir taş seçilmişse, hamle yapmayı dene
        if self.selected:
            result = self._move(row, col, all_moves)  # _move fonksiyonuna all_moves bilgisini geçebiliriz.
            if not result:
                self.selected = None
                self.select(row, col)
                return

        piece = self.board.get_piece(row, col)
        # Seçilen taş, o oyuncunun taşlarından biri olmalı ve hamle listesinde yer almalı
        if piece != 0 and piece.color == self.turn and piece in all_moves:
            self.selected = piece
            self.valid_moves = all_moves[piece]
            return True

        return False
                
    def get_all_pieces(self, color):
        pieces = []
        for row in range(ROWS):      # ROWS, board satır sayısını temsil ediyor
            for col in range(COLS):  # COLS, board sütun sayısını temsil ediyor
                piece = self.board.get_piece(row, col)
                if piece != 0 and piece.color == color:
                    pieces.append(piece)
        return pieces

    def _move(self, row, col, all_moves):
        if not self.selected or (row, col) not in self.valid_moves:
            return False

        # 1. Hamle öncesi durumu kaydet
        was_king = self.selected.king

        # 2. Taşı yeni konumuna taşı (Burada dama olabilir)
        self.board.move(self.selected, row, col)

        # 3. Hamle sonrası durumu kontrol et
        is_now_king = self.selected.king

        skipped = self.valid_moves[(row, col)]  # Yenen taşları al

        if skipped:
            self.board.remove(skipped)  # Yenen taşları tahtadan kaldır

            # --- KRİTİK DÜZELTME BURASI ---
            # Eğer taş bu hamlede dama olduysa (ve önceden değildiyse),
            # taş yeme hakkı olsa bile tur BİTER.
            if not was_king and is_now_king:
                self.change_turn()
                return True
            # ------------------------------

            # Hamleden sonra tekrar yeme şansı var mı?
            self.valid_moves = self.board.get_valid_moves(self.selected)

            # Eğer tekrar yeme imkanı varsa ve mevcut hamle en fazla taş yiyen hamlelerden biriyse devam et
            max_captures = 0
            if all_moves:
                max_captures = max(len(v) for v in all_moves.values() if v)

            if any(self.valid_moves.values()) and len(skipped) == max_captures:
                return True  # Oyuncu tekrar oynamalı

        # Eğer tekrar yeme hamlesi yoksa tur değiştir
        self.change_turn()
        return True