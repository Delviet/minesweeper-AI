import numpy as np
from numba import jit
from scipy.ndimage import convolve

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
        self.player_field = np.full(self.typical_shape, '*')
        
    def fully_generate(self, x_point, y_point):
        self.mines = np.zeros(self.typical_shape, dtype = 'bool')
        self.place_mines(x_point, y_point)
        
        self.find_values()
        
    def place_mines(self, x_point, y_point):
        self.amount_of_mines = place_mines(self.height, self.width, self.alpha, self.mines, x_point, y_point)
    
    def find_values(self):
        mask = np.ones((3,3))
        mask[1][1] = 0
        self.values = convolve(self.mines.astype(np.uint8), mask, mode='constant', cval=0.0)
    
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
                if (x > 0 and y > 0): result += self.enter(x-1,y-1)
                if (x > 0 and y < self.height - 1): result += self.enter(x-1,y+1)
                if (x < self.width - 1 and y > 0): result += self.enter(x+1,y-1)
                if (x < self.width - 1 and y < self.height - 1): result += self.enter(x+1,y+1)
            
        return result

    
@jit(nopython=True)
def place_mines(height, width, alpha, mines, x, y):
    amount_of_mines = int(np.round(height * width * alpha))
    positions = np.arange(height * width)
    positions = np.delete(positions, y*width + x)
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
        self.closed = self.height * self.width
        self.game_runned = True
        self.field_generated = False
        self.won = False
        self.lose = False
    
    def do_step(self, x, y):
        if not self.field_generated:
            self.pg.fully_generate(x, y)
            self.mines_amount = self.pg.amount_of_mines

            self.field_generated = True
        if self.game_runned:
            result = self.pg.enter(int(x), int(y))
            if result == -1:
                self.fail_game()
            else:
                self.closed -= result
        if self.closed == self.mines_amount:
            self.win_game()
        
    def fail_game(self):
        self.game_runned = False
        self.lose = True
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
            frames = np.empty((self.height * self.width, 5, 5), dtype = 'str')
            new_df = np.zeros((self.height + 4, self.width + 4), dtype = 'str')
            new_df[2:-2, 2:-2] = self.pg.player_field
            for y in range(self.height):
                for x in range(self.width):
                    frames[y * self.width + x] = new_df[y:y+5, x:x+5].reshape(1, 5, 5)
            return frames