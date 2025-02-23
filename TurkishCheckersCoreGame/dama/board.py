import pygame
from .constants import DARK_BROWN, ROWS, COLS, LIGHT_BROWN, SQUARE_SIZE, BLACK, WHITE
from .piece import Piece

class Board:
    def __init__(self):
        self.board = []
        self.black_left = self.white_left = 16
        self.black_kings = self.white_kings = 0
        self.create_board()
        
    def draw_squares(self, win):
        win.fill(DARK_BROWN)
        for row in range(ROWS):
            for col in range(row % 2, COLS, 2):
                pygame.draw.rect(win, LIGHT_BROWN, (row*SQUARE_SIZE, col *SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))
    
    def move(self, piece, row, col):
        self.board[piece.row][piece.col], self.board[row][col] = self.board[row][col], self.board[piece.row][piece.col]
        piece.move(row, col)
        
        #renklere göre dama olmayı ayır
        if row == ROWS - 1 or row == 0:
            piece.make_king()
            if piece.color == BLACK:
                self.black_kings += 1
            else:
                self.white_kings += 1
    
    def get_piece(self, row, col):
        return self.board[row][col]
    
    def create_board(self):
        for row in range(ROWS):
            self.board.append([])
            for col in range(COLS):
                if row > 0 and row < 3:
                    self.board[row].append(Piece(row, col, BLACK))
                elif row > 4 and row < 7:
                    self.board[row].append(Piece(row, col, WHITE))
                else:
                    self.board[row].append(0)
        
    def draw(self, win):
        self.draw_squares(win)
        for row in range(ROWS):
            for col in range(COLS):
                piece = self.board[row][col]
                if piece != 0:
                    piece.draw(win)
    
    def remove(self, pieces):
        for piece in pieces:
            self.board[piece.row][piece.col] = 0
            if piece != 0:
                if piece.color == WHITE:
                    self.white_left -= 1
                else:
                    self.black_left -=1
    
    def winner(self):
        if self.white_left <= 0:
            return BLACK
        elif self.black_left <= 0:
            return WHITE
        
        return None
    
      def evaluate(self):
        return self.black_left - self.white_left + (self.black_kings * 2 - self.white_kings * 2)

    
    def get_valid_moves(self, piece):
        moves = {}
        row = piece.row
        col = piece.col
        is_king = piece.king
        color = piece.color

        if is_king:
            # Dama: tüm yönlerde tarama yap
            moves.update(self._traverse_left_horizontal(col, color, row, king=True))
            moves.update(self._traverse_right_horizontal(col, color, row, king=True))
            moves.update(self._traverse_forward_vertical(row, color, col, king=True))
            moves.update(self._traverse_backward_vertical(row, color, col, king=True))
        else:
            # Normal taşlar: beyaz için yukarı, siyah için aşağı + yatay
            moves.update(self._traverse_left_horizontal(col, color, row, king=False))
            moves.update(self._traverse_right_horizontal(col, color, row, king=False))
            if color == WHITE:
                moves.update(self._traverse_forward_vertical(row, color, col, king=False))
            else:  # BLACK
                moves.update(self._traverse_backward_vertical(row, color, col, king=False))
        
        return moves


    def _traverse_left_horizontal(self, start_column, color, row, skipped=[], king=False):
        moves = {}
        last = []  # Yakalanacak rakip taş (henüz yakalanmadıysa boş kalır)
        for c in range(start_column - 1, -1, -1):
            current = self.board[row][c]
            if current == 0:
                # Boş kare bulundu
                if skipped and not last:
                    # Halihazırda yakalama zincirindeysek ama bu yönde yakalanacak taş yoksa durdur.
                    break
                if last:
                    # Yakalama hamlesi: rakip taşın hemen arkasındaki boş kare
                    moves[(row, c)] = last + skipped
                    new_skipped = last + skipped
                    # Recursive arama: ek yakalama seçeneklerini ekle
                    if king:
                        # Dama tüm yönlerde recursive çağrı yapabilir
                        moves.update(self._traverse_left_horizontal(c, color, row, skipped=new_skipped, king=king))
                        moves.update(self._traverse_forward_vertical(row, color, c, skipped=new_skipped, king=king))
                        moves.update(self._traverse_backward_vertical(row, color, c, skipped=new_skipped, king=king))
                    else:
                        # Normal taşlar için yalnızca izin verilen yönlerde recursive çağrı (beyaz için yukarı, BLACK için aşağı)
                        if color == WHITE:
                            moves.update(self._traverse_left_horizontal(c, color, row, skipped=new_skipped, king=king))
                            moves.update(self._traverse_forward_vertical(row, color, c, skipped=new_skipped, king=king))
                        else:  # BLACK
                            moves.update(self._traverse_left_horizontal(c, color, row, skipped=new_skipped, king=king))
                            moves.update(self._traverse_backward_vertical(row, color, c, skipped=new_skipped, king=king))
                    # Normal taşlar sadece tek boş kare üzerinden atlayabildiğinden döngüyü kırıyoruz.
                    if not king:
                        break
                else:
                    # Yakalama zinciri yoksa, normal hamle (sadece 1 kare ilerleme)
                    if not skipped:
                        moves[(row, c)] = []
                    # Normal taş hareketinde ilk boş kareden sonra dur.
                    if not king:
                        break
            elif current != 0:
                if current.color == color:
                    # Kendi taşımız engel oluşturur
                    break
                else:
                    # Rakip taş bulundu
                    if last:
                        # Aynı yönde zaten bir rakip taş belirlendiyse, ardışık iki rakibi atlamak yasak
                        break
                    last = [current]
        return moves


    def _traverse_right_horizontal(self, start_column, color, row, skipped=[], king=False):
        moves = {}
        last = []
        for c in range(start_column + 1, COLS):
            current = self.board[row][c]
            if current == 0:
                if skipped and not last:
                    break
                if last:
                    moves[(row, c)] = last + skipped
                    new_skipped = last + skipped
                    if king:
                        moves.update(self._traverse_right_horizontal(c, color, row, skipped=new_skipped, king=king))
                        moves.update(self._traverse_forward_vertical(row, color, c, skipped=new_skipped, king=king))
                        moves.update(self._traverse_backward_vertical(row, color, c, skipped=new_skipped, king=king))
                    else:
                        if color == WHITE:
                            moves.update(self._traverse_right_horizontal(c, color, row, skipped=new_skipped, king=king))
                            moves.update(self._traverse_forward_vertical(row, color, c, skipped=new_skipped, king=king))
                        else:
                            moves.update(self._traverse_right_horizontal(c, color, row, skipped=new_skipped, king=king))
                            moves.update(self._traverse_backward_vertical(row, color, c, skipped=new_skipped, king=king))
                    if not king:
                        break
                else:
                    if not skipped:
                        moves[(row, c)] = []
                    if not king:
                        break
            elif current != 0:
                if current.color == color:
                    break
                else:
                    if last:
                        break
                    last = [current]
        return moves


    def _traverse_forward_vertical(self, start_row, color, col, skipped=[], king=False):
        moves = {}
        last = []
        for r in range(start_row - 1, -1, -1):
            current = self.board[r][col]
            if current == 0:
                if skipped and not last:
                    break
                if last:
                    moves[(r, col)] = last + skipped
                    new_skipped = last + skipped
                    if king:
                        moves.update(self._traverse_left_horizontal(col, color, r, skipped=new_skipped, king=king))
                        moves.update(self._traverse_right_horizontal(col, color, r, skipped=new_skipped, king=king))
                        moves.update(self._traverse_forward_vertical(r, color, col, skipped=new_skipped, king=king))
                    else:
                        # Normal taş: beyaz taş için sadece ileri (yukarı) yön recursive
                        moves.update(self._traverse_left_horizontal(col, color, r, skipped=new_skipped, king=king))
                        moves.update(self._traverse_right_horizontal(col, color, r, skipped=new_skipped, king=king))
                        moves.update(self._traverse_forward_vertical(r, color, col, skipped=new_skipped, king=king))
                    if not king:
                        break
                else:
                    if not skipped:
                        moves[(r, col)] = []
                    if not king:
                        break
            elif current != 0:
                if current.color == color:
                    break
                else:
                    if last:
                        break
                    last = [current]
        return moves


    def _traverse_backward_vertical(self, start_row, color, col, skipped=[], king=False):
        moves = {}
        last = []
        for r in range(start_row + 1, ROWS):
            current = self.board[r][col]
            if current == 0:
                if skipped and not last:
                    break
                if last:
                    moves[(r, col)] = last + skipped
                    new_skipped = last + skipped
                    if king:
                        moves.update(self._traverse_left_horizontal(col, color, r, skipped=new_skipped, king=king))
                        moves.update(self._traverse_right_horizontal(col, color, r, skipped=new_skipped, king=king))
                        moves.update(self._traverse_backward_vertical(r, color, col, skipped=new_skipped, king=king))
                    else:
                        # Normal taş: BLACK için sadece geri (aşağı) yön recursive
                        moves.update(self._traverse_left_horizontal(col, color, r, skipped=new_skipped, king=king))
                        moves.update(self._traverse_right_horizontal(col, color, r, skipped=new_skipped, king=king))
                        moves.update(self._traverse_backward_vertical(r, color, col, skipped=new_skipped, king=king))
                    if not king:
                        break
                else:
                    if not skipped:
                        moves[(r, col)] = []
                    if not king:
                        break
            elif current != 0:
                if current.color == color:
                    break
                else:
                    if last:
                        break
                    last = [current]
        return moves


    
    """
    def get_valid_moves(self, piece):
        moves = {}
        row = piece.row
        col = piece.col
        is_king = piece.king
        color = piece.color

        # Eğer taş dama ise tüm yönler; normal taşlarda ise renk bazlı yönler kontrol edilir.
        if is_king:
            moves.update(self._traverse_left_horizontal(col, color, row, king=True))
            moves.update(self._traverse_right_horizontal(col, color, row, king=True))
            moves.update(self._traverse_forward_vertical(row, color, col, king=True))
            moves.update(self._traverse_backward_vertical(row, color, col, king=True))
        else:
            moves.update(self._traverse_left_horizontal(col, color, row, king=False))
            moves.update(self._traverse_right_horizontal(col, color, row, king=False))
            if color == WHITE:
                moves.update(self._traverse_forward_vertical(row, color, col, king=False))
            else:
                moves.update(self._traverse_backward_vertical(row, color, col, king=False))
        
        return moves


    def _traverse_left_horizontal(self, start_column, color, row, skipped=[], king=False):
        moves = {}
        last = []  # Bu değişkende, o yönde yakalanacak (henüz atlanmamış) rakip taş tutulur.
        for c in range(start_column - 1, -1, -1):
            current = self.board[row][c]
            if current == 0:
                # Boş kare bulundu
                if skipped and not last:
                    # Eğer halihazırda bir atlama zinciri içerisindeysek fakat bu yöne ait rakip taş yoksa, taramayı durdur.
                    break
                if last:
                    # Bu boş kare, yakalanan rakip taşın arkasındadır.
                    moves[(row, c)] = last + skipped
                    # Yakalama yaptıktan sonra, yeni konumdan (row, c) ileri ek yakalama seçeneklerine bakıyoruz.
                    if king:
                        moves.update(self._traverse_left_horizontal(c, color, row, skipped=last+skipped, king=king))
                        moves.update(self._traverse_forward_vertical(row, color, c, skipped=last+skipped, king=king))
                        moves.update(self._traverse_backward_vertical(row, color, c, skipped=last+skipped, king=king))
                    else:
                        if color == WHITE:
                            moves.update(self._traverse_left_horizontal(c, color, row, skipped=last+skipped, king=king))
                            moves.update(self._traverse_forward_vertical(row, color, c, skipped=last+skipped, king=king))
                        else:  # BLACK
                            moves.update(self._traverse_left_horizontal(c, color, row, skipped=last+skipped, king=king))
                            moves.update(self._traverse_backward_vertical(row, color, c, skipped=last+skipped, king=king))
                else:
                    # Normal (atlamasız) hamle: fakat eğer zaten bir yakalama zincirindeysek, normal hamle eklenmez.
                    if not skipped:
                        moves[(row, c)] = []
                # Normal taşlarda boş kareye ulaştığında devam etmez; dama için (king) boş kareler devam ettirilebilir.
                if not king or last:
                    break
            elif current.color == color:
                # Kendi taşımıza çarptı: yol kapalı.
                break
            else:
                # Rakip taş bulundu
                if last:
                    # Aynı yönde zaten bir rakip taş belirlenmişse, ardışık iki rakip taş atlanamaz.
                    break
                last = [current]
        return moves


    def _traverse_right_horizontal(self, start_column, color, row, skipped=[], king=False):
        moves = {}
        last = []
        for c in range(start_column + 1, COLS):
            current = self.board[row][c]
            if current == 0:
                if skipped and not last:
                    break
                if last:
                    moves[(row, c)] = last + skipped
                    if king:
                        moves.update(self._traverse_right_horizontal(c, color, row, skipped=last+skipped, king=king))
                        moves.update(self._traverse_forward_vertical(row, color, c, skipped=last+skipped, king=king))
                        moves.update(self._traverse_backward_vertical(row, color, c, skipped=last+skipped, king=king))
                    else:
                        if color == WHITE:
                            moves.update(self._traverse_right_horizontal(c, color, row, skipped=last+skipped, king=king))
                            moves.update(self._traverse_forward_vertical(row, color, c, skipped=last+skipped, king=king))
                        else:
                            moves.update(self._traverse_right_horizontal(c, color, row, skipped=last+skipped, king=king))
                            moves.update(self._traverse_backward_vertical(row, color, c, skipped=last+skipped, king=king))
                else:
                    if not skipped:
                        moves[(row, c)] = []
                if not king or last:
                    break
            elif current.color == color:
                break
            else:
                if last:
                    break
                last = [current]
        return moves


    def _traverse_forward_vertical(self, start_row, color, col, skipped=[], king=False):
        moves = {}
        last = []
        for r in range(start_row - 1, -1, -1):
            current = self.board[r][col]
            if current == 0:
                if skipped and not last:
                    break
                if last:
                    moves[(r, col)] = last + skipped
                    if king:
                        moves.update(self._traverse_left_horizontal(col, color, r, skipped=last+skipped, king=king))
                        moves.update(self._traverse_right_horizontal(col, color, r, skipped=last+skipped, king=king))
                        moves.update(self._traverse_forward_vertical(r, color, col, skipped=last+skipped, king=king))
                    else:
                        if color == WHITE:
                            moves.update(self._traverse_left_horizontal(col, color, r, skipped=last+skipped, king=king))
                            moves.update(self._traverse_right_horizontal(col, color, r, skipped=last+skipped, king=king))
                            moves.update(self._traverse_forward_vertical(r, color, col, skipped=last+skipped, king=king))
                        else:
                            moves.update(self._traverse_left_horizontal(col, color, r, skipped=last+skipped, king=king))
                            moves.update(self._traverse_right_horizontal(col, color, r, skipped=last+skipped, king=king))
                            moves.update(self._traverse_backward_vertical(r, color, col, skipped=last+skipped, king=king))
                else:
                    if not skipped:
                        moves[(r, col)] = []
                if not king or last:
                    break
            elif current.color == color:
                break
            else:
                if last:
                    break
                last = [current]
        return moves


    def _traverse_backward_vertical(self, start_row, color, col, skipped=[], king=False):
        moves = {}
        last = []
        for r in range(start_row + 1, ROWS):
            current = self.board[r][col]
            if current == 0:
                if skipped and not last:
                    break
                if last:
                    moves[(r, col)] = last + skipped
                    if king:
                        moves.update(self._traverse_left_horizontal(col, color, r, skipped=last+skipped, king=king))
                        moves.update(self._traverse_right_horizontal(col, color, r, skipped=last+skipped, king=king))
                        moves.update(self._traverse_backward_vertical(r, color, col, skipped=last+skipped, king=king))
                    else:
                        if color == WHITE:
                            moves.update(self._traverse_left_horizontal(col, color, r, skipped=last+skipped, king=king))
                            moves.update(self._traverse_right_horizontal(col, color, r, skipped=last+skipped, king=king))
                            moves.update(self._traverse_forward_vertical(r, color, col, skipped=last+skipped, king=king))
                        else:
                            moves.update(self._traverse_left_horizontal(col, color, r, skipped=last+skipped, king=king))
                            moves.update(self._traverse_right_horizontal(col, color, r, skipped=last+skipped, king=king))
                            moves.update(self._traverse_backward_vertical(r, color, col, skipped=last+skipped, king=king))
                else:
                    if not skipped:
                        moves[(r, col)] = []
                if not king or last:
                    break
            elif current.color == color:
                break
            else:
                if last:
                    break
                last = [current]
        return moves
    """
#############################################################################
    """
    def get_valid_moves(self, piece):
        moves = {}
        row = piece.row
        col = piece.col

        # Yatay (sola/sağa) ve dikey (ileri) fonksiyonları çağır
        if piece.color == WHITE or piece.king:
            # Beyaz taşlar ve kral taşlar için: yukarı + yatay
            moves.update(self._traverse_left_horizontal(col, piece.color, row))
            moves.update(self._traverse_right_horizontal(col, piece.color, row))
            moves.update(self._traverse_forward_vertical(row, piece.color, col))
    
        if piece.color == BLACK or piece.king:
            # Siyah taşlar ve kral taşlar için: aşağı + yatay
            moves.update(self._traverse_left_horizontal(col, piece.color, row))
            moves.update(self._traverse_right_horizontal(col, piece.color, row))
            moves.update(self._traverse_backward_vertical(row, piece.color, col))  # İleri=aşağı (opsiyonel)

        return moves        
    
    def _traverse_left_horizontal(self, start_column, color, row, skipped=[]):
        moves = {}
        last = []
        self.row = row  # Mevcut taşın satırı
    
        # Sütunları sola tarama (start_column'dan 0'a kadar azalan)
        for c in range(start_column - 1, -1, -1):  # -1 adımıyla 0 dahil
            current = self.board[row][c]
            if current == 0:
                if skipped and not last:
                    break
                if last:
                    # Atlama yapıldı: moves[(r, col)] = last + skipped
                    moves[(row, c)] = last + skipped
                    # Çoklu zıplama için diğer yönlere bak
                    if color == WHITE:
                        moves.update(self._traverse_left_horizontal(c, color, row, skipped=last+skipped))
                         #moves.update(self._traverse_right_horizontal(c, color, row, skipped=last+skipped))
                        moves.update(self._traverse_forward_vertical(row, color, c, skipped=last+skipped))
                    else:
                        moves.update(self._traverse_left_horizontal(c, color, row, skipped=last+skipped))
                         #moves.update(self._traverse_right_horizontal(c, color, row, skipped=last+skipped))
                        moves.update(self._traverse_backward_vertical(row, color, c, skipped=last+skipped))
                else:
                    moves[(row, c)] = last + skipped
                break
            elif current.color == color:
                break
            else:
                if last:
                    break
                last = [current]
    
        return moves
    
    def _traverse_right_horizontal(self, start_column, color, row, skipped=[]):
        moves = {}
        last = []
        # Sabit bir satır kullan (mevcut taşın satırı: row)
        self.row = row  # Mevcut taşın satırını al (örnek varsayım)
        
        # Sütunları sağa doğru taramak için döngü (start_column'dan COLS'a kadar)
        for c in range(start_column + 1, COLS):
            current = self.board[row][c]
            if current == 0:
                # Boş hücre: Hamle ekle (skipped + last)
                if skipped and not last:
                    break
                if last:
                    # Atlama yapıldı: moves[(r, col)] = last + skipped
                    moves[(row, c)] = last + skipped
                    # Çoklu zıplama için diğer yönlere bak
                    if color == WHITE:
                         #moves.update(self._traverse_left_horizontal(c, color, row, skipped=last+skipped))
                        moves.update(self._traverse_right_horizontal(c, color, row, skipped=last+skipped))
                        moves.update(self._traverse_forward_vertical(row, color, c, skipped=last+skipped))
                    else:
                         #moves.update(self._traverse_left_horizontal(c, color, row, skipped=last+skipped))
                        moves.update(self._traverse_right_horizontal(c, color, row, skipped=last+skipped))
                        moves.update(self._traverse_backward_vertical(row, color, c, skipped=last+skipped))
                else:
                    moves[(row, c)] = last + skipped
                break
            elif current.color == color:
                # Kendi renginde taş: Engel
                break
            else:
                # Rakip taşı: Atlama yap (last'e ekle)
                if last:
                    break
                last = [current]
    
        return moves
    
    
    def _traverse_forward_vertical(self, start_row, color, col, skipped=[]):
        moves = {}
        last = []
        self.col = col  # Mevcut taşın sütunu
    
        # Satırları yukarı (ileri) tarama (start_row'dan 0'a azalan)
        for r in range(start_row - 1, -1, -1):
            current = self.board[r][col]
            if current == 0:
                if skipped and not last:
                    break
                if last:
                    # Atlama yapıldı: moves[(r, col)] = last + skipped
                    moves[(r, col)] = last + skipped
                    # Çoklu zıplama için diğer yönlere bak
                    moves.update(self._traverse_left_horizontal(col, color, r, skipped=last+skipped))
                    moves.update(self._traverse_right_horizontal(col, color, r, skipped=last+skipped))
                    moves.update(self._traverse_forward_vertical(r, color, col, skipped=last+skipped))
                else:
                    # Normal hamle
                    moves[(r, col)] = last + skipped
                break
            elif current.color == color:
                break
            else:
                if last:
                    break
                last = [current]
    
        return moves
    
    def _traverse_backward_vertical(self, start_row, color, col, skipped=[]):
        moves = {}
        last = []
        self.col = col  # Taşın mevcut sütunu
    
        # Aşağı doğru tarama (start_row'dan ROWS'a kadar)
        for r in range(start_row + 1, ROWS):
            current = self.board[r][col]
            if current == 0:
                if skipped and not last:
                    break
                if last:
                    # Atlama yapıldı: moves[(r, col)] = last + skipped
                    moves[(r, col)] = last + skipped
                    # Çoklu zıplama için diğer yönlere bak
                    moves.update(self._traverse_left_horizontal(col, color, r, skipped=last+skipped))
                    moves.update(self._traverse_right_horizontal(col, color, r, skipped=last+skipped))
                    moves.update(self._traverse_backward_vertical(r, color, col, skipped=last+skipped))  # Tekrar aşağı
                else:
                    # Normal hamle
                    moves[(r, col)] = last + skipped
                break
            elif current.color == color:
                break
            else:
                if last:
                    break
                last = [current]
    
        return moves
        """