import time
import random

def q_learn(model, time_limit):
    learning_rate = 0.25
    discount_factor = 0.9
    epsilon = 0.1
    playbook_size = model.offensive_playbook_size()

    q_values = {}
    for a in range(playbook_size):
        for r in range(3):
            for c in range(3):
                q_values[(r,c,a)] = 0

    # q_values = [[[0 for _ in range(3)] for _ in range(3)] for _ in range(playbook_size)]

    def find_bucket(state, action):
        yards_to_td, downs_left, yards_to_1d, ticks = state
        x = yards_to_1d/downs_left if downs_left != 0 else float('inf')
        y = yards_to_td/ticks if ticks != 0 else float('inf')

        bucket_x = None
        bucket_y = None

        if x < 1:
            bucket_x = 0
        elif x < 2:
            bucket_x = 1
        else:
            bucket_x = 2
        
        if y < 2:
            bucket_y = 0
        elif y < 4:
            bucket_y = 1
        else:
            bucket_y = 2
        
        return (bucket_x, bucket_y, action)

    def q_value(state, action):
        return q_values[find_bucket(state, action)] 
        # return q_values[bucket_x][bucket_y][action] 

    def q_update(state, action, reward, next_state):
        # bucket_x, bucket_y, action = find_bucket(state, action)
        current_q = q_value(state, action)
        max_next_q = max(q_value(next_state, a) for a in range(model.offensive_playbook_size()))
        td_target = reward + discount_factor * max_next_q
        td_error = td_target - current_q
        q_values[find_bucket(state, action)] += learning_rate * td_error
        # q_values[bucket_x][bucket_y][action] += learning_rate * td_error

    def policy(position):
        if random.random() < epsilon:
            return random.randint(0, playbook_size - 1)
        else:
            q_vals = [q_value(state, a) for a in range(playbook_size)]
            return max(range(playbook_size), key=lambda a: q_vals[a])

    end_time = time.time() + time_limit    

    while time.time() < end_time:
        state = model.initial_position()
        while not model.game_over(state):
            action = policy(state)
            new_state = model.result(state, action)[0]
            # reward = yards_gained
            reward = 0
            if not model.game_over(new_state):
                reward = 0
            elif model.win(new_state):
                reward = 1
            else:
                reward = -1
            # if turnover:
            #     reward = -100  # Large negative reward for turnover
            # else:
            #     reward = yards_gained  # Reward based on yards gained
            q_update(state, action, reward, new_state)
            state = new_state
    
    return policy
