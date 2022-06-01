import numpy as np
import torch
import random


def validation(env, student, opponent, n_games, epoch, loss, arg, random_player=False):
    wins = 0
    for i in range(n_games):
        observation = env.reset()
        done = False
        while not done:
            with torch.no_grad():
                if env.turn == 1:
                    action = student.choose_action(observation)
                    observation_, reward, done, info = env.step(action)
                    if reward == 1:
                        wins += 1
                    observation = observation_
                elif not done:
                    action = opponent.choose_action(observation)
                    observation_, reward, done, info = env.step(action)
                    observation = observation_
    win_pct = wins / n_games
    if random_player:
        print("Performance against random player:")
    else:
        print("Performance against teacher:")
    print(f'Average win percantage after {epoch} games: {win_pct:.1%}. Epsilon: {student.epsilon:.2f}, Loss: {loss}')
    if win_pct > arg.win:
        return True
    else:
        return False


def set_random_seed(id=42):
    random.seed(id)
    torch.manual_seed(id)
    torch.cuda.manual_seed(id)
    np.random.seed(id)
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(id)
    rng = np.random.RandomState(id)
