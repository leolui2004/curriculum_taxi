# training loop
while time_step <= max_training_timesteps:
    
    state = env.reset()
    state_list = state_decode(state)
    
    current_ep_reward = 0

    for t in range(1, max_ep_len+1):
        
        action = ppo_agent.select_action(state_list)
        state, reward, done, _ = env.step(action)
        state_list = state_decode(state)
        reward_, done_stage, done_ = done_stage_process(state_list, reward, done_stage, done)
        
        ppo_agent.buffer.rewards.append(reward_)
        ppo_agent.buffer.is_terminals.append(done)
        
        time_step +=1
        current_ep_reward += reward_

        if time_step % update_timestep == 0:
            ppo_agent.update()

        if has_continuous_action_space and time_step % action_std_decay_freq == 0:
            ppo_agent.decay_action_std(action_std_decay_rate, min_action_std)

        if time_step % log_freq == 0:

            log_avg_reward = log_running_reward / log_running_episodes
            log_avg_reward = round(log_avg_reward, 4)

            log_f.write('{},{},{}\n'.format(i_episode, time_step, log_avg_reward))
            log_f.flush()

            log_running_reward = 0
            log_running_episodes = 0

        if time_step % print_freq == 0:

            print_avg_reward = print_running_reward / print_running_episodes
            print_avg_reward = round(print_avg_reward, 2)

            print("Eps. : {} \t Timestep : {} \t Avg. Reward : {} \t Goal : {}".format(i_episode, time_step, print_avg_reward, print_running_goal))

            print_running_reward = 0
            print_running_episodes = 0
            print_running_goal = 0
            
        if time_step % save_model_freq == 0:
            print("--------------------------------------------------------------------------------------------")
            print("saving model at : " + checkpoint_path)
            ppo_agent.save(checkpoint_path)
            print("model saved")
            print("Elapsed Time  : ", datetime.now().replace(microsecond=0) - start_time)
            print("--------------------------------------------------------------------------------------------")
            
        if done_ == 1 or done:
            break

    print_running_reward += current_ep_reward
    print_running_episodes += 1
    if done_ == 1:
        print_running_goal += 1

    log_running_reward += current_ep_reward
    log_running_episodes += 1

    i_episode += 1