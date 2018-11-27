import torch
import numpy as np

class Binary_World(object):
    def __init__(self, grid_size, noise, gamma, dim_states=9, prob_blue = 0.9):
        # Actions: (up, right, down, left)

        assert(grid_size>0 and (0 <= gamma <=1) and (0 <= noise <=1))
        self.grid_size = grid_size
        self.actions = (-grid_size, +1 , +grid_size, -1)
        self.n_actions = len(self.actions)
        self.n_states = grid_size**2
        self.noise = noise
        self.gamma = gamma
        self.transition_probability = torch.zeros((self.n_states,self.n_actions,self.n_states))
        self.feature_matrix = torch.zeros((self.n_states,dim_states))

        #generate the transition matrix
        for state in range(self.n_states):
            for action in range(self.n_actions):
                a = self.actions[action]
                other_actions = self.actions[0:action] + self.actions[action + 1: self.n_actions]
                if not self._is_corner_action(state,a):
                    self.transition_probability[state,action,state + a] += 1-self.noise
                else:
                    self.transition_probability[state,action,state] += 1-self.noise

                for o in other_actions:
                    if not self._is_corner_action(state,o):
                        self.transition_probability[state,action,state + o] += self.noise / (self.n_actions - 1)
                    else:
                        self.transition_probability[state,action,state] += self.noise / (self.n_actions - 1)

        #generate the feature matrix
        #0: red, 1: blue
        for state in range(self.n_states):
            self.feature_matrix[state,0] = np.random.binomial(1,prob_blue)


        for state in range(self.n_states):
            neighbours = (state - self.grid_size, state - self.grid_size + 1, state + 1, state + self.grid_size + 1, state + self.grid_size, state + self.grid_size -1, state - 1, state - self.grid_size -1)
            for n in neighbours:
                if n in range(0,grid_size**2):
                    self.feature_matrix[state,neighbours.index(n)+1] = self.feature_matrix[n,0]
                else:
                    self.feature_matrix[state,neighbours.index(n)+1] = 0

        self.reward_vector = torch.zeros(self.n_states)
        for s in range(self.n_states):
            self.reward_vector[s] += self.reward(s)




    def _is_corner_action(self, state_index, action):

        if action == -self.grid_size and (state_index in range(self.grid_size)):
            return True
        if action == self.grid_size and (state_index in range(self.grid_size * (self.grid_size -1), self.grid_size**2)):
            return True
        if action == +1 and (state_index in range(self.grid_size-1,self.grid_size**2,self.grid_size)):
            return True
        if action == -1 and (state_index in range(0, self.grid_size * (self.grid_size-1) + 1, self.grid_size)):
            return True
        else:
            return False


    def reward(self,state):
        blue_count = torch.sum(self.feature_matrix[state,:])
        if blue_count == 4:
            return 1
        if blue_count == 5:
            return -1
        else:
            return 0


    def state_value(self, policy, epsilon=0.001, horizon=1000, reward_vector=None):
        v_func = torch.zeros(self.n_states)
        if reward_vector is None:
            reward_vector = self.reward_vector

        delta = 0
        for h in range(horizon):

            for s in range(self.n_states):
                v_temp = v_func[s].item()

                _, a = policy[s,:].max(0)
                v_new = torch.sum(self.transition_probability[s,a.long(),:] @ (reward_vector + self.gamma * v_func))
                delta = max(delta, abs(v_temp - v_new.item()))

                v_func[s] = v_new

            if delta < epsilon:
                print("Breaking Horizon:{}".format(h))
                break

        return v_func

    def optimum_state_value(self, epsilon=0.001, horizon=1000, reward_vector=None):
        v_func = torch.zeros(self.n_states)
        if reward_vector is None:
            reward_vector = self.reward_vector

        delta = 0
        for h in range(horizon):

            for s in range(self.n_states):
                m = float("-inf")
                for a in range(self.n_actions):
                    m = max(m, (self.transition_probability[s,a,:] @ (reward_vector + self.gamma * v_func)).item())

                delta = max(delta, abs(v_func[s].item() - m))
                v_func[s] = m
            if delta < epsilon:
                print("Breaking Horizon:{}".format(h))
                break

        return v_func

    def get_policy(self, epsilon=0.001, values=None, deterministic=True):
        if values is None:
            v_func = self.optimum_state_value(epsilon=epsilon)
        else:
            v_func = values

        policy = torch.zeros((self.n_states,self.n_actions))
        if deterministic is True:
            for s in range(self.n_states):
                action_values = torch.zeros(self.n_actions)
                for a in range(self.n_actions):
                    action_values[a] += torch.sum(self.transition_probability[s,a,:] * (self.reward_vector + self.gamma * v_func))
                    #for s_ in range(self.n_states):
                    #    action_value += self.transition_probability[s,a,s_] * (self.reward(s_) + self.gamma * v_func[s_])

                _, best_action= torch.max(action_values,0)
                policy[s,best_action.long()] = 1

        else:
            for s in range(self.n_states):
                for a in range(self.n_actions):
                    policy[s,a] = torch.sum(self.transition_probability[s, a, :] * (self.reward_vector + self.gamma * v_func))

            policy = torch.exp(policy)
            for s in range(self.n_states):
                policy[s,:] = policy[s,:] / torch.sum(policy[s,:])

        return policy


    def generate_trajectories(self, n_trajectory, horizon, policy, starting_probs=None):

        trajectories = torch.zeros((n_trajectory,horizon,3))

        for n in range(n_trajectory):
            if starting_probs is None:
                s_0 = np.random.choice(self.n_states)

            else:
                s_0 = np.random.choice(self.n_states, p=starting_probs)

            s = s_0

            for t in range(horizon):
                action_index = np.random.choice(self.n_actions,p=policy[s,:])
                transition_prob = self.transition_probability[s,action_index,:]
                next_state = np.random.choice(self.n_states,p=transition_prob)
                reward = self.reward(next_state)
                trajectories[n,t,0] = s
                trajectories[n, t, 1] = action_index
                trajectories[n, t, 2] = reward
                s=next_state


        return trajectories

class Linear_World(object):
    def __init__(self, grid_size, noise, gamma, reward_parameter, dim_states=9, prob_blue=0.9):
        # Actions: (up, right, down, left)

        assert (grid_size > 0 and (0 <= gamma <= 1) and (0 <= noise <= 1))
        self.grid_size = grid_size
        self.actions = (-grid_size, +1, +grid_size, -1)
        self.n_actions = len(self.actions)
        self.n_states = grid_size ** 2
        self.noise = noise
        self.gamma = gamma
        self.transition_probability = torch.zeros((self.n_states, self.n_actions, self.n_states))
        self.feature_matrix = torch.zeros((self.n_states, dim_states))
        self.reward_parameter = reward_parameter

        # generate the transition matrix
        for state in range(self.n_states):
            for action in range(self.n_actions):
                a = self.actions[action]
                other_actions = self.actions[0:action] + self.actions[action + 1: self.n_actions]
                if not self._is_corner_action(state, a):
                    self.transition_probability[state, action, state + a] += 1 - self.noise
                else:
                    self.transition_probability[state, action, state] += 1 - self.noise

                for o in other_actions:
                    if not self._is_corner_action(state, o):
                        self.transition_probability[state, action, state + o] += self.noise / (self.n_actions - 1)
                    else:
                        self.transition_probability[state, action, state] += self.noise / (self.n_actions - 1)

        # generate the feature matrix
        # 0: red, 1: blue
        for state in range(self.n_states):
            self.feature_matrix[state, 0] = np.random.binomial(1, prob_blue)

        for state in range(self.n_states):
            neighbours = (state - self.grid_size, state - self.grid_size + 1, state + 1, state + self.grid_size + 1,
                          state + self.grid_size, state + self.grid_size - 1, state - 1, state - self.grid_size - 1)
            for n in neighbours:
                if n in range(0, grid_size ** 2):
                    self.feature_matrix[state, neighbours.index(n) + 1] = self.feature_matrix[n, 0]
                else:
                    self.feature_matrix[state, neighbours.index(n) + 1] = 0

        self.reward_vector = torch.zeros(self.n_states)
        for s in range(self.n_states):
            self.reward_vector[s] += self.reward(s)

    def _is_corner_action(self, state_index, action):

        if action == -self.grid_size and (state_index in range(self.grid_size)):
            return True
        if action == self.grid_size and (
            state_index in range(self.grid_size * (self.grid_size - 1), self.grid_size ** 2)):
            return True
        if action == +1 and (state_index in range(self.grid_size - 1, self.grid_size ** 2, self.grid_size)):
            return True
        if action == -1 and (state_index in range(0, self.grid_size * (self.grid_size - 1) + 1, self.grid_size)):
            return True

    def reward(self, state):
        return self.reward_parameter @ self.feature_matrix[state, :]

    def state_value(self, policy, epsilon=0.001, horizon=1000, reward_vector=None):
        v_func = torch.zeros(self.n_states)
        if reward_vector is None:
            reward_vector = self.reward_vector

        for h in range(horizon):
            delta = 0
            for s in range(self.n_states):
                v_temp = v_func[s]

                _, a = policy[s,].max(0)
                v_new = torch.sum(
                    self.transition_probability[s, a.long(), :] @ (reward_vector + self.gamma * v_func))
                delta = max(delta, abs(v_temp - v_new))

                v_func[s] = v_new

            if delta < epsilon:
                print("Breaking Horizon:{}".format(h))
                break

        return v_func

    def optimum_state_value(self, epsilon=0.001, horizon=1000, reward_vector=None):
        v_func = torch.zeros(self.n_states)
        if reward_vector is None:
            reward_vector = self.reward_vector

        for h in range(horizon):
            delta = 0
            for s in range(self.n_states):
                m = float("-inf")
                for a in range(self.n_actions):
                    m = max(m, (self.transition_probability[s, a, :] @ (reward_vector + self.gamma * v_func)))

                delta = max(delta, abs(v_func[s] - m))
                v_func[s] = m
            if delta < epsilon:
                print("Breaking Horizon:{}".format(h))
                break

        return v_func

    def get_policy(self, epsilon=0.001, values=None, deterministic=True):
        if values is None:
            v_func = self.optimum_state_value(epsilon=epsilon)
        else:
            v_func = values

        policy = torch.zeros((self.n_states, self.n_actions))
        if deterministic is True:
            for s in range(self.n_states):
                best_action_value = float("-inf")
                action = None
                for a in range(self.n_actions):
                    action_value = 0
                    for s_ in range(self.n_states):
                        action_value += self.transition_probability[s, a, s_] * (
                        self.reward(s_) + self.gamma * v_func[s_])

                    if best_action_value < action_value:
                        best_action_value = action_value
                        action = a
                policy[s, action] = 1

        else:
            for s in range(self.n_states):
                for a in range(self.n_actions):
                    action_value = 0
                    for s_ in range(self.n_states):
                        action_value += self.transition_probability[s, a, s_] * (
                        self.reward(s_) + self.gamma * v_func[s_])
                    policy[s, a] = action_value

            policy = torch.exp(policy)
            for s in range(self.n_states):
                policy[s, :] = policy[s, :] / torch.sum(policy[s, :])

        return policy

    def generate_trajectories(self, n_trajectory, horizon, policy, starting_probs=None):

        trajectories = torch.zeros((n_trajectory, horizon, 3))

        for n in range(n_trajectory):
            if starting_probs is None:
                s_0 = np.random.choice(self.n_states)

            else:
                s_0 = np.random.choice(self.n_states, p=starting_probs)

            s = s_0

            for t in range(horizon):
                action_index = np.random.choice(self.n_actions, p=policy[s, :])
                transition_prob = self.transition_probability[s, action_index, :]
                next_state = np.random.choice(self.n_states, p=transition_prob)
                reward = self.reward(next_state)
                trajectories[n, t, 0] = s
                trajectories[n, t, 1] = action_index
                trajectories[n, t, 2] = reward
                s = next_state

        return trajectories

#M = Binary_World(3,0.0,0.9,prob_blue=0.8)
#deterministic_policy = M.get_policy()
#optim_value_func = M.optimum_state_value()
#print(M.reward_vector)
#print(optim_value_func)












