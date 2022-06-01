from Brain import Brain
import numpy as np
import torch
import random


class Agent(object):
    def __init__(self, gamma, epsilon, input_dims, batch_size, n_actions, arg,
                 mem_size=1_000_000, eps_end=0.1, eps_dec=0.9999):
        self.lr = arg.lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_end = eps_end
        self.eps_dec = eps_dec
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.action_space = [i for i in range(n_actions)]
        self.mem_size = mem_size  # we need a memory to store experiences and randommly sample over them
        self.mem_counter = 0
        self.Q_eval = Brain(arg)
        self.Q_target = Brain(arg)
        self.Q_target.load_state_dict(self.Q_eval.state_dict())
        self.Q_target.eval()
        self.state_memory = np.zeros((self.mem_size, *input_dims))
        self.new_state_memory = np.zeros((self.mem_size, *input_dims))
        self.action_memory = np.zeros((self.mem_size, self.n_actions), dtype=np.uint8)
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.uint8)  # sequence of done flags
        self.allowed_next_move_memory = np.zeros(self.mem_size)
        self.discard_value = -1e7

    def store_transition(self, state, action, reward, state_, terminal):
        index = self.mem_counter % self.mem_size
        self.state_memory[index] = state.reshape(-1)
        actions = np.zeros(self.n_actions)
        actions[action] = 1.0  # one hot encoding of actions
        self.action_memory[index] = actions
        self.reward_memory[index] = reward
        self.terminal_memory[index] = not terminal
        self.new_state_memory[index] = state_.reshape(-1)
        self.mem_counter += 1

    def choose_action(self, observation):
        allowed_actions = [i for i in range(observation.shape[1]) if (observation[:, i] == 0).sum() > 0]
        rand = np.random.random()
        if rand < self.epsilon:
            action = np.random.choice(allowed_actions)
            return action
        else:
            with torch.no_grad():
                A = self.Q_eval.forward(observation)
                actions = A.squeeze(0)[allowed_actions]
                action_index = torch.argmax(actions).item()
                action = allowed_actions[action_index]
        return action

    @staticmethod
    def choose_random_action(observation):
        valid_cols = [i for i in range(observation.shape[1]) if (observation[:, i] == 0).sum() > 0]
        random_action = random.choice(valid_cols)
        return random_action

    def learn(self):
        if self.mem_counter > self.batch_size:
            self.Q_eval.optimizer.zero_grad()

            max_mem = self.mem_counter if self.mem_counter < self.mem_size \
                else self.mem_size
            batch = np.random.choice(max_mem, self.batch_size)
            batch_index = np.arange(self.batch_size, dtype=np.int32)

            state_batch = self.state_memory[batch]
            action_batch = self.action_memory[batch]
            action_values = np.array(self.action_space, dtype=np.int32)
            action_indices = np.dot(action_batch, action_values)
            reward_batch = self.reward_memory[batch]
            terminal_batch = self.terminal_memory[batch]
            new_state_batch = self.new_state_memory[batch]

            # V - A
            # V_s, A_s = self.Q_eval(state_batch)
            # V_s_, A_s_ = self.Q_eval(new_state_batch)
            # V_s, A_s, V_s_, A_s_ = V_s.to(self.Q_eval.device), A_s.to(self.Q_eval.device), \
            #                        V_s_.to(self.Q_eval.device), A_s_.to(self.Q_eval.device)
            #
            # q_pred = torch.add(V_s, (A_s - A_s.mean(dim=1, keepdim=True)))[batch_index, action_indices]
            # q_next = torch.add(V_s_, (A_s_ - A_s_.mean(dim=1, keepdim=True)))
            # move_validity = torch.Tensor(new_state_batch[:, :self.n_actions] == 0).to(self.Q_eval.device)
            # discard_values = self.discard_value * torch.ones([self.batch_size, self.n_actions]).to(self.Q_eval.device)
            # q_next = torch.where(move_validity == 1., q_next, discard_values)
            # max_actions = torch.argmax(q_next, dim=1)

            # Get predicted q values for the actions that were taken
            q_pred = self.Q_eval.forward(state_batch).to(self.Q_eval.device)
            q_pred = q_pred[batch_index, action_indices]
            # Replace -1 and 1 for new_state_batch
            new_state_batch *= -1.
            q_eval = self.Q_eval.forward(new_state_batch).to(self.Q_eval.device)

            # Get target q values for the actions that were taken
            move_validity = torch.Tensor(new_state_batch[:, :self.n_actions] == 0).to(self.Q_eval.device)
            discard_values = self.discard_value * torch.ones([self.batch_size, self.n_actions]).to(self.Q_eval.device)
            q_next = self.Q_target.forward(new_state_batch).to(self.Q_eval.device)
            q_eval = torch.where(move_validity == 1., q_eval, discard_values)
            max_actions = torch.argmax(q_eval, dim=1)

            reward_batch = torch.Tensor(reward_batch).to(self.Q_eval.device)
            terminal_batch = torch.Tensor(terminal_batch).to(self.Q_eval.device)

            # Using minimax algorithm
            q_target = reward_batch + self.gamma * (-q_next[batch_index, max_actions]) * terminal_batch
            loss = self.Q_eval.loss(q_pred, q_target.detach()).to(self.Q_eval.device)
            loss.backward()

            # Update epsilon and optimizer
            self.epsilon = max(self.epsilon * self.eps_dec, self.eps_end)
            self.Q_eval.optimizer.step()

            return loss