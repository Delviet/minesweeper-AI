import numpy as np
from numba import jit

class Playground:
    
    
    def __init__(self, height, width, alpha = 0.2):
        
        assert (alpha < 1) and (alpha > 0)
        assert (height >= 3) and (type(height) == int)
        assert (width >= 3) and (type(width) == int)
        
        self.height = height
        self.width = width
        self.square = self.height * self.width
        self.alpha = alpha
        
        self.typical_shape = (self.height, self.width)
        
        self.mines = np.zeros(self.typical_shape, dtype = 'bool')
        self.place_mines()
        
        self.find_values()
        
        self.player_field = np.full(self.typical_shape, '*')
        
    def place_mines(self):
        self.amount_of_mines = place_mines(self.height, self.width, self.alpha, self.mines)
    
    def find_values(self):
        mask = np.ones((3,3))
        mask[1][1] = 0
        self.values = convolve(self.mines, mask, mode='constant', cval=0.0)
    
    def show(self):
        print()
        for y in range(self.height):
            for x in range(self.width):
                print(self.player_field[y, x], end="")
            print()
        print()
        
    def enter(self, x, y):
        
        result = 0
        
        if self.player_field[y, x] == '*':
            if self.mines[y, x]:
                self.player_field[y, x] = '!'
                return -1
            
            result += 1
            self.player_field[y, x] = str(self.values[y, x])
            
            if self.values[y, x] == 0:
                if x > 0: result += self.enter(x-1,y)
                if y > 0: result += self.enter(x,y-1)
                if x < self.width - 1: result += self.enter(x+1,y)
                if y < self.height - 1: result += self.enter(x,y+1)
            
        return result

    
@jit(nopython=True)
def place_mines(height, width, alpha, mines):
    amount_of_mines = int(np.round(height * width * alpha))
    positions = np.arange(height * width)
    np.random.shuffle(positions)
    for pos in positions[:amount_of_mines]:
        mines[pos // width, pos % width] = True
        
    return amount_of_mines


class Game:
    
    def __init__(self, height, width, alpha = 0.2):
        self.height = height
        self.width = width
        self.alpha = alpha
        
    def start(self, mode = 'player'):
        assert mode == 'player' or mode == 'bot'
        self.mode = mode
        self.pg = Playground(self.height, self.width, self.alpha)
        self.mines_amount = self.pg.amount_of_mines
        self.closed = self.height * self.width
        self.game_runned = True
        self.won = False
    
    def do_step(self, x, y):
        if self.game_runned:
            result = self.pg.enter(int(x), int(y))
            if result == -1:
                self.fail_game()
            else:
                self.closed -= result
        self.pg.show()
        if self.closed == self.mines_amount:
            self.win_game()
        
    def fail_game(self):
        self.game_runned = False
        if self.mode == 'player':
            print("You have failed!")
        
    def win_game(self):
        self.game_runned = False
        self.won = True
        if self.mode == 'player':
            print("You have won!")
        
    def show(self):
        if self.mode == 'player':
            self.pg.show()
        else:
            return self.pg.player_field

