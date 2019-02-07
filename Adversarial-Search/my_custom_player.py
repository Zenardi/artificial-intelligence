
from sample_players import DataPlayer
from isolation.isolation import _HEIGHT, _WIDTH, _SIZE
from collections import defaultdict, Counter
import random



class CustomPlayer(DataPlayer):
    """ Implement your own agent to play knight's Isolation

    The get_action() method is the only required method for this project.
    You can modify the interface for get_action by adding named parameters
    with default values, but the function MUST remain compatible with the
    default interface.

    **********************************************************************
    NOTES:
    - The test cases will NOT be run on a machine with GPU access, nor be
      suitable for using any other machine learning techniques.

    - You can pass state forward to your agent on the next turn by assigning
      any pickleable object to the self.context attribute.
    **********************************************************************
    """
    def get_action(self, state):
        """ Employ an adversarial search technique to choose an action
        available in the current state calls self.queue.put(ACTION) at least

        This method must call self.queue.put(ACTION) at least once, and may
        call it as many times as you want; the caller will be responsible
        for cutting off the function after the search time limit has expired.

        See RandomPlayer and GreedyPlayer in sample_players for more examples.

        **********************************************************************
        NOTE: 
        - The caller is responsible for cutting off search, so calling
          get_action() from your own code will create an infinite loop!
          Refer to (and use!) the Isolation.play() function to run games.
        **********************************************************************
        """
        # TODO: Replace the example implementation below with your own search
        #       method by combining techniques from lecture
        #
        # EXAMPLE: choose a random move without any search--this function MUST
        #          call self.queue.put(ACTION) at least once before time expires
        #          (the timer is automatically managed for you)
        if state.ply_count < 2:
            # Without Opening Book (random opeining moves)
            # self.queue.put(random.choice(state.actions()))
            
            # With Opening Book            
            try:
                table = {game_state: max(action, key = action.get) \
                            for game_state, action in self.data}
            except:
                table = {}
                
            if state in table:
                if table[state] in state.actions():
                    self.queue.put(table[state])
                else:
                    self.queue.put(random.choice(state.actions()))
            else:
                self.queue.put(random.choice(state.actions()))
        else:            
            # No Iterative Deepening
            # move, score = self.alpha_beta(state, 3)
            # self.queue.put(move)
                        
            # Iterative Deepening
            best_move = random.choice(state.actions())
            best_score = float('-inf')
            depth_limit = 10
            
            for depth in range(1, depth_limit + 1): 
                move, score = self.alpha_beta(state, depth)
                                  
                if score > best_score:
                    best_move = move
                    best_score = score
                    
                self.queue.put(best_move)
        return
    
    def alpha_beta(self, state, depth):
        
        def min_value(state, alpha, beta, depth):
            if state.terminal_test():
                return state.utility(self.player_id)   
            if depth <= 0: 
                return self.score(state, self.player_id)
            
            value = float("inf")
            
            for action in state.actions():
                value = min(value, max_value(state.result(action), alpha, beta, depth - 1))
                if value <= alpha:
                    return value
                beta = min(beta, value)
            
            return value
    
        def max_value(state, alpha, beta, depth):
            if state.terminal_test():
                return state.utility(self.player_id)
            if depth <= 0: 
                    return self.score(state, self.player_id)
            
            value = float("-inf")
            
            for action in state.actions():
                value = max(value, min_value(state.result(action), alpha, beta, depth - 1))
                if value >= beta:
                    return value
                alpha = max(alpha, value)
                
            return value
    
        alpha = float("-inf")
        beta = float("inf")
        best_score = float("-inf")
        best_move = None
        
        for action in state.actions():
            value = min_value(state.result(action), alpha, beta, depth - 1)
            alpha = max(alpha, value)
            
            if value > best_score:
                best_score = value
                best_move = action
                
        return best_move, best_score

    def minimax(self, state, depth):

        def min_value(state, depth):
            if state.terminal_test(): 
                return state.utility(self.player_id)
            if depth <= 0: 
                return self.score(state, self.player_id)
            value = float("inf")
            for action in state.actions():
                value = min(value, max_value(state.result(action), depth - 1))
            return value

        def max_value(state, depth):
            if state.terminal_test(): 
                return state.utility(self.player_id)
            if depth <= 0: 
                return self.score(state, self.player_id)
            value = float("-inf")
            for action in state.actions():
                value = max(value, min_value(state.result(action), depth - 1))
            return value

        best_score = float("-inf")
        best_move = None
        
        for action in state.actions():
            value = min_value(state.result(action), depth - 1)
            
            if value > best_score:
                best_score = value
                best_move = action
                
        return best_move, best_score
    
    
    def loc_to_xy(self, loc):
        '''Converts a location (int) to a pair of x, y coordinates'''
        return float(loc % (_WIDTH + 2)), float(loc // (_WIDTH + 2))
    
    def distance(self, loc_a, loc_b):
        '''Returns the Euclidean distance between two locations'''
        a_x, a_y = self.loc_to_xy(loc_a)
        b_x, b_y = self.loc_to_xy(loc_b)
        return ((a_x - b_x)**2 + (a_y -b_y)**2)**0.5
        
    def distance_to_center(self, loc):
        '''Returns the distance from the location to the center of board'''
        center = _SIZE // 2
        return self.distance(loc, center)
    
    def in_corner(self, loc, epsilon = 0.001):
        '''Returns True if location is a corner of board'''
        return self.distance_to_center(loc) > (self.distance_to_center(0) - epsilon) and \
               self.distance_to_center(loc) < (self.distance_to_center(0) + epsilon)
    
    def corners(self, size = _SIZE):
        '''Locations of the 4 corners'''
        return set([loc for loc in range(size) if self.in_corner(loc)])
    
    def distance_to_corner(self, loc):
        '''Distance to closest corner'''
        return min([self.distance(loc, corner) for corner in self.corners()]) 
    
    
    
    dist_to_center = \
        {0: 6.4031242374328485, 1: 5.656854249492381, 2: 5.0, 3: 4.47213595499958, 
         4: 4.123105625617661, 5: 4.0, 6: 4.123105625617661, 7: 4.47213595499958, 
         8: 5.0, 9: 5.656854249492381, 10: 6.4031242374328485, 13: 5.830951894845301, 
         14: 5.0, 15: 4.242640687119285, 16: 3.605551275463989, 17: 3.1622776601683795, 
         18: 3.0, 19: 3.1622776601683795, 20: 3.605551275463989, 21: 4.242640687119285, 
         22: 5.0, 23: 5.830951894845301, 26: 5.385164807134504, 27: 4.47213595499958,
         28: 3.605551275463989, 29: 2.8284271247461903, 30: 2.23606797749979, 31: 2.0, 
         32: 2.23606797749979, 33: 2.8284271247461903, 34: 3.605551275463989, 
         35: 4.47213595499958, 36: 5.385164807134504, 39: 5.0990195135927845, 
         40: 4.123105625617661, 41: 3.1622776601683795, 42: 2.23606797749979, 
         43: 1.4142135623730951, 44: 1.0, 45: 1.4142135623730951, 46: 2.23606797749979, 
         47: 3.1622776601683795, 48: 4.123105625617661, 49: 5.0990195135927845, 52: 5.0, 
         53: 4.0, 54: 3.0, 55: 2.0, 56: 1.0, 57: 0.0, 58: 1.0, 59: 2.0, 60: 3.0, 61: 4.0, 
         62: 5.0, 65: 5.0990195135927845, 66: 4.123105625617661, 67: 3.1622776601683795, 
         68: 2.23606797749979, 69: 1.4142135623730951, 70: 1.0, 71: 1.4142135623730951, 
         72: 2.23606797749979, 73: 3.1622776601683795, 74: 4.123105625617661, 
         75: 5.0990195135927845, 78: 5.385164807134504, 79: 4.47213595499958, 
         80: 3.605551275463989, 81: 2.8284271247461903, 82: 2.23606797749979, 83: 2.0, 
         84: 2.23606797749979, 85: 2.8284271247461903, 86: 3.605551275463989, 
         87: 4.47213595499958, 88: 5.385164807134504, 91: 5.830951894845301, 92: 5.0, 
         93: 4.242640687119285, 94: 3.605551275463989, 95: 3.1622776601683795, 96: 3.0, 
         97: 3.1622776601683795, 98: 3.605551275463989, 99: 4.242640687119285, 100: 5.0, 
         101: 5.830951894845301, 104: 6.4031242374328485, 105: 5.656854249492381, 106: 5.0, 
         107: 4.47213595499958, 108: 4.123105625617661, 109: 4.0, 110: 4.123105625617661, 
         111: 4.47213595499958, 112: 5.0, 113: 5.656854249492381, 114: 6.4031242374328485}
    
    
    
    def score(self, state, player_id, heuristic = 2):
        
        if state.utility(player_id) > 0:
            return float('inf')
        if state.utility(player_id) < 0:
            return float('-inf')
        
        own_loc = state.locs[player_id]
        opp_loc = state.locs[1 - player_id]
        
        own_liberties = state.liberties(own_loc)
        opp_liberties = state.liberties(opp_loc)
        
        if heuristic == 1:         
            return len(own_liberties) - len(opp_liberties)
        
        if heuristic == 2:
            return (self.dist_to_center[opp_loc] + 1) * len(own_liberties) - \
                   (self.dist_to_center[own_loc] + 1) * len(opp_liberties)      