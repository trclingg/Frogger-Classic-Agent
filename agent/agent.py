import json
import os
import random

from .state import State


class Q_State(State):
    '''Augments the game state with Q-learning information'''

    def __init__(self, string):
        super().__init__(string)

        # key stores the state's key string (see notes in _compute_key())
        self.key = self._compute_key()

    def _compute_key(self):
        '''
        Returns a key used to index this state.

        The key should reduce the entire game state to something much smaller
        that can be used for learning. When implementing a Q table as a
        dictionary, this key is used for accessing the Q values for this
        state within the dictionary.
        '''

        # this simple key uses the 3 object characters above the frog
        # and combines them into a key string
        return ''.join([
            self.get(self.frog_x - 1, self.frog_y - 1) or '_',
            self.get(self.frog_x, self.frog_y - 1) or '_',
            self.get(self.frog_x + 1, self.frog_y - 1) or '_',
            self.get(self.frog_x - 1, self.frog_y) or '_',
            self.get(self.frog_x + 1, self.frog_y) or '_',
            self.get(self.frog_x - 1, self.frog_y + 1) or '_',
            self.get(self.frog_x, self.frog_y + 1) or '_',
            self.get(self.frog_x + 1, self.frog_y + 1) or '_'
        ])

    def reward(self):
        '''Returns a reward value for the state.'''

        if self.at_goal:
            return self.score
        elif self.is_done:
            return -10
        else:
            return 0


class Agent:

    def __init__(self, train=None):
        self.learning_rate = 0.1
        self.discount_factor = 0.9
        self.prev_state = None
        self.prev_action = None
        # train is either a string denoting the name of the saved
        # Q-table file, or None if running without training
        self.train = train

        # q is the dictionary representing the Q-table
        self.q = {}

        # name is the Q-table filename
        # (you likely don't need to use or change this)
        self.name = train or 'q'

        # path is the path to the Q-table file
        # (you likely don't need to use or change this)
        self.path = os.path.join(os.path.dirname(
            os.path.realpath(__file__)), 'train', self.name + '.json')

        self.load()
        

    def load(self):
        '''Loads the Q-table from the JSON file'''
        try:
            with open(self.path, 'r') as f:
                self.q = json.load(f)
            if self.train:
                print('Training {}'.format(self.path))
            else:
                print('Loaded {}'.format(self.path))
        except IOError:
            if self.train:
                print('Training {}'.format(self.path))
            else:
                raise Exception('File does not exist: {}'.format(self.path))
        return self

    def save(self):
        '''Saves the Q-table to the JSON file'''
        with open(self.path, 'w') as f:
            json.dump(self.q, f)
        return self

    

    def initialize_q_table(self):
       
        if self.prev_state is not None:
            current_state = Q_State(self.prev_state)
            key = current_state.key

            if key not in self.q:
                self.q[key] = {'u': 0, 'd': 0, 'l': 0, 'r': 0, '_': 0}
                if self.prev_action is not None:
                    # Choose a random action for the uninitialized state
                    self.q[key][self.prev_action] = random.uniform(0, 1)


    def choose_action(self, state_string):
        '''
        Returns the action to perform.

        This is the main method that interacts with the game interface:
        given a state string, it should return the action to be taken
        by the agent.

        The initial implementation of this method is simply a random
        choice among the possible actions. You will need to augment
        the code to implement Q-learning within the agent.
        '''

        current_state = Q_State(state_string)

        # check if its the first move or not
        if self.prev_state is None or self.prev_action is None:
            chosen_action = random.choice(State.ACTIONS)       
            
        # exploration vs exploitation
        exploration_rate = 0.1
        if random.uniform(0, 2) < exploration_rate:
            chosen_action = random.choice(State.ACTIONS)
        else:
            # exploitation: choose action w max Q-value
            q_values = self.q.get(current_state.key, {})
            chosen_action = max(q_values, key=q_values.get, default=None)

        self.initialize_q_table()
        
        if self.prev_state is not None and self.prev_action is not None:
            reward = current_state.reward()  
            next_state_string = state_string  
            self.update_q_table(reward, next_state_string)
                    
        
        self.prev_state = state_string
        self.prev_action = chosen_action
            

        return chosen_action
    
    
    def update_q_table(self, reward, next_state_string):
        current_state = Q_State(self.prev_state)
        next_state = Q_State(next_state_string)
        # alpha: learning rate
        alpha = self.learning_rate
        # gamma: discount factor
        gamma = self.discount_factor

       
        # q-value for the current state and action
        current_q_value = self.q.get(current_state.key, {}).get(self.prev_action, 0)

        # max q-value for the next state
        max_next_q_value = max(self.q.get(next_state.key, {}).values(), default=0)
        

        # update q-value using the formula
        new_q_value = (1 - alpha) * current_q_value + alpha * (reward + gamma * max_next_q_value)

        # update in q-table
        if current_state.key not in self.q:
            self.q[current_state.key] = {}
        self.q[current_state.key][self.prev_action] = new_q_value
        self.save()
        