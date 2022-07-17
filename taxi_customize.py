def state_decode(i):
    out = []
    out.append(i % 4)
    i = i // 4
    out.append(i % 5)
    i = i // 5
    out.append(i % 5)
    i = i // 5
    out.append(i)
    return list(reversed(out))

def done_stage_process(state_list, reward, done_stage, done):
    done_ = 0
    reward_ = 0
    
    if done_stage == 0: # go to pick up
        if state_list[2] == 0:
            if state_list[0] == 0 and state_list[1] == 0:
                reward_ = 150
                done_stage = 1
        elif state_list[2] == 1:
            if state_list[0] == 0 and state_list[1] == 4:
                reward_ = 150
                done_stage = 1
        elif state_list[2] == 2:
            if state_list[0] == 4 and state_list[1] == 0:
                reward_ = 150
                done_stage = 1
        elif state_list[2] == 3:
            if state_list[0] == 4 and state_list[1] == 3:
                reward_ = 150
                done_stage = 1
    
    elif done_stage == 1: # pick up
        if state_list[2] == 4:
            done_stage = 2
            reward_ = 300
    
    elif done_stage == 2: # go to drop off
        if state_list[3] == 0:
            if state_list[0] == 0 and state_list[1] == 0:
                reward_ = 450
                done_stage = 3
        elif state_list[3] == 1:
            if state_list[0] == 0 and state_list[1] == 4:
                reward_ = 450
                done_stage = 3
        elif state_list[3] == 2:
            if state_list[0] == 4 and state_list[1] == 0:
                reward_ = 450
                done_stage = 3
        elif state_list[3] == 3:
            if state_list[0] == 4 and state_list[1] == 3:
                reward_ = 450
                done_stage = 3
    
    elif done_stage == 3: # drop off
        if reward_ == 20:
            reward_ = 600
            done_ = 1
    
    if reward_ == -10:
        reward_ = -5
    
    return reward, done_stage, done_

def action_curiosity(curiosity, action_space, action):
    if random.random() > curiosity:
        return action
    else:
        return random.randint(0, action_space - 1)