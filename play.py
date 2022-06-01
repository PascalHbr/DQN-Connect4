import torch
from Agent import Agent
from config import parse_args
from env import Connect4Env


def play(arg):
    device = torch.device(f'cuda:{arg.gpu}' if torch.cuda.is_available() else 'cpu')

    # Load env and teacher
    env = Connect4Env()
    teacher = Agent(gamma=0.99, epsilon=0., batch_size=512, n_actions=7,
                    input_dims=[42], arg=arg)
    teacher.Q_eval.load_state_dict(torch.load('parameters/' + arg.name + '_parameters.pth', map_location=device))

    play = True
    while play:
        # Play a Game
        observation = env.reset()
        env.render()
        done = False
        while not done:
            if env.turn == 1:
                action = teacher.choose_action(observation)
                observation_, reward, done, info = env.step(action)
                observation = observation_
                env.render()
            elif not done:
                allowed_actions = [i for i in range(observation.shape[1]) if (observation[:, i] == 0).sum() > 0]
                action = input('Make a move: ')
                while not action.isnumeric() or int(action) not in allowed_actions:
                    action = input('Not a valid move!!! Make another move: ')
                observation_, reward, done, info = env.step(int(action))
                observation = observation_
                env.render()


if __name__ == '__main__':
    arg = parse_args()
    play(arg)