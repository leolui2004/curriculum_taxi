# Using Curriculum Reinforcement Learning to Solve Taxi Problem

## Introduction

Reinforcement learning in the past is weak to solve problems with multiple stages, using the example of Taxi-v3, normal actor-critic method will not work without using replay queue (DQN will work in this case). By introducing curriculum reinforcement learning, the model can learn to complete the goal stage by stage, at the end to finish the whole task. I have also tried to add some curiosity to see if there are any effects to the effectiveness of training.

## Taxi-v3

https://github.com/openai/gym/blob/master/gym/envs/toy_text/taxi.py

The game is simple, a taxi starts in a random position in a 5 x 5 field, the taxi needs to go to a position randomly chosen by the environment each time to pick up the passenger, and then go to another position to drop off the passenger.

There 500 states, 6 actions (left, right, up, down, pick up, drop off), with only one goal but indeed 4 stages (go to the pick up position, pick up, go to the drop off position, drop off). With a +20 reward for achieving the goal, -10 for a wrong pick up or drop off action, -1 for each move per timestep.

## Methodology

The Taxi-v3 provided by OpenAI is a wrapped environment, to use it as a curriculum learning and also easier for the algorithm to learn, I need to modify the state, action, reward, done status.

Customizing State

```
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
```

Customizing Action

```
def action_curiosity(curiosity, action_space, action):
    if random.random() > curiosity:
        return action
    else:
        return random.randint(0, action_space - 1)
```

Customizing Reward and decision for Done

```def done_stage_process(state_list, reward, done_stage, done):
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
```

For the reinforcement learning part, I used PPO with slight modification, the reference code can be found here.

https://github.com/nikhilbarhate99/PPO-PyTorch

## Result

### Run 1: Train as usual (Baseline)

![image](https://github.com/leolui2004/curriculum_taxi/blob/main/r1.png)

### Run 2: Train as usual, with 1% of curiosity

![image](https://github.com/leolui2004/curriculum_taxi/blob/main/r2.png)

The total reward stopped increasing until -200 and never reached a goal, this may be because the model received -20 for wrong pick up or drop off making it tended to just moving around until the game end (maximum timestep 200).

### Run 3: Goal is set to go to the pick up location only (stage 1 of 4), reward for illegal pick up or drop off is set to -5, goal for the task is 150

![image](https://github.com/leolui2004/curriculum_taxi/blob/main/r3.png)

Although the task is reduced to only the first stage, the model was able to learn to achieve positive cumulative reward and also increasing achievement of the goal.

### Run 4: Goal is set to go to the pick up location and after finished pick up only (stage 2 of 4), reward for illegal pick up or drop off is set to -5, goal for the task is 250

![image](https://github.com/leolui2004/curriculum_taxi/blob/main/r4.png)

### Run 5: Goal is set to go to the pick up location and after finished pick up only (stage 2 of 4), reward for illegal pick up or drop off is set to -5, goal for the task is 300, with an intermediate goal 150 if the taxi successfully reached the pick up location

![image](https://github.com/leolui2004/curriculum_taxi/blob/main/r5.png)

![image](https://github.com/leolui2004/curriculum_taxi/blob/main/r5_.png)

Both of them are successfully learned to complete the task, and the difference between setting up an intermediate goal does not show a lot of difference here.

### Run 6: Goal is set to go to the pick up location, pick up and go to the drop off location only (stage 3 of 4), reward for illegal pick up or drop off is set to -5, goal for the task is 450, with an intermediate goal 150 if the taxi successfully reached the pick up location, and an intermediate goal 300 if the taxi successfully picked up the passenger

![image](https://github.com/leolui2004/curriculum_taxi/blob/main/r6.png)

### Run 7: Goal is set to same as the original goal (stage 4 of 4), reward for illegal pick up or drop off is set to -5, goal for the task is 600, with an intermediate goal 150 if the taxi successfully reached the pick up location, an intermediate goal 300 if the taxi successfully picked up the passenger, and an intermediate goal 450 if the taxi successfully reached the drop off location

![image](https://github.com/leolui2004/curriculum_taxi/blob/main/r7.png)

### Run 8: Same with Run 7 but with 1% of curiosity

![image](https://github.com/leolui2004/curriculum_taxi/blob/main/r8.png)

### Run 9: Same with Run 8 but the learning rate decrease by 50% each 200,000 timesteps

![image](https://github.com/leolui2004/curriculum_taxi/blob/main/r9.png)

The model successfully learned until stage 3 but failed to learn until stage 4, with even curiosity added, the model at first tried to complete the task a few times but at the end it still failed to learn just like the result in Run 1. And the result is also not much improved even with the learning rate is scheduled to decrease.

### Run 10: Same reward with Run 7 but using curriculum learning, the last 200,000 timesteps are with original learning rate * 0.1

![image](https://github.com/leolui2004/curriculum_taxi/blob/main/r10.png)

The model learned significantly fast for the first 3 stages but failed for the last stage, note that for the first 3 stages, each stage the model first had a drop on cumulative reward but then it quickly learned to complete a more difficult task.

### Run 11: Same reward with Run 7 but using curriculum learning, the training timesteps are doubled for 3rd stage and 4th stage, and the last stage are with original learning rate * 0.1

![image](https://github.com/leolui2004/curriculum_taxi/blob/main/r11.png)

![image](https://github.com/leolui2004/curriculum_taxi/blob/main/r11_.png)

![image](https://github.com/leolui2004/curriculum_taxi/blob/main/r11__.png)

The model can learn successfully although the result is not that brilliant. A simple comparison with Run 6 will also find that the one using curriculum learning can learn with almost the same speed compared with that without using curriculum learning.
