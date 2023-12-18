import time, random

def q_learn(model, time_limit):
    def find_bucket(state, action):
        yards_to_td, downs_left, yards_to_1d, ticks = state
        x = yards_to_1d/downs_left if downs_left != 0 else float('inf')
        y = yards_to_td/ticks if ticks != 0 else float('inf')

        bucket_x, bucket_y = 0, 0

        if x < 2.25:
            bucket_x += 0
        elif x < 4.4:
            bucket_x += 1
        else:
            bucket_x += 2
        
        if y < 2.1:
            bucket_y += 0
        elif y < 4.15:
            bucket_y += 1
        else:
            bucket_y += 2
        
        return (bucket_x, bucket_y, action)

    def q_update(state, action, reward, next_state):
        # bucket_x, bucket_y, action = find_bucket(state, action)
        state_key = find_bucket(state, action)
        if model.game_over(next_state):
            q_values[state_key] += (alpha_vals[state_key] * (reward - q_values[state_key]))
        else:
            next_action = get_optimal_action(next_state)
            new_state_key = find_bucket(next_state, next_action)
            td_target = reward + discount_factor*q_values[new_state_key]
            td_error = td_target - q_values[state_key] 
            q_values[state_key] += (alpha_vals[state_key] * td_error)
        alpha_vals[state_key] *= 0.999
        
    def get_optimal_action(state):
        q_vals = [q_values[find_bucket(state, a)] for a in range(playbook_size)]
        return max(range(playbook_size), key=lambda a: q_vals[a])

    def get_action(state):
        if random.random() < epsilon:
            return random.randint(0, playbook_size - 1)
        else:
            return get_optimal_action(state)

    learning_rate = 0.25
    discount_factor = 0.99
    epsilon = 0.15
    playbook_size = model.offensive_playbook_size()

    q_values = {}
    alpha_vals = {}
    for a in range(playbook_size):
        for r in range(3):
            for c in range(3):
                q_values[(r,c,a)] = 0
                alpha_vals[(r,c,a)] = learning_rate

    end_time = time.time() + time_limit    

    while time.time() < end_time:
        state = model.initial_position()
        while not model.game_over(state):
            action = get_action(state)
            new_state = model.result(state, action)[0]
            reward = 0
            if not model.game_over(new_state):
                reward = 0
            elif model.win(new_state):
                reward = 1
            else:
                reward = -1
            q_update(state, action, reward, new_state)
            state = new_state
    
    return get_optimal_action
