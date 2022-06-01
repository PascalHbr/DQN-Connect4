import torch
from Agent import Agent
from config import parse_args
from copy import deepcopy
from env import Connect4Env
from utils import validation, set_random_seed


def main(arg):
    # Set random seeds
    set_random_seed()

    # Hyperparameters
    tau = arg.tau

    # Set up device / env / agents
    device = torch.device(f'cuda:{arg.gpu}' if torch.cuda.is_available() else 'cpu')
    env = Connect4Env()
    student = Agent(gamma=arg.gamma, epsilon=arg.eps_start, batch_size=arg.bn, n_actions=7,
                    input_dims=[42], arg=arg, mem_size=arg.mem_size, eps_end=arg.eps_end)
    teacher = Agent(gamma=arg.gamma, epsilon=arg.eps_teacher, batch_size=arg.bn, n_actions=7,
                    input_dims=[42], arg=arg)
    random_player = Agent(gamma=0.99, epsilon=1., batch_size=arg.bn, n_actions=7,
                          input_dims=[42], arg=arg)

    if arg.load:
        student.Q_eval.load_state_dict(torch.load('/parameters/' + arg.name + '_parameters.pth', map_location=device))
        student.Q_target.load_state_dict(torch.load('/parameters/' + arg.name + '_parameters.pth', map_location=device))
        teacher.Q_eval.load_state_dict(torch.load('/parameters/' + arg.name + '_parameters.pth', map_location=device))
        teacher.Q_target.load_state_dict(torch.load('/parameters/' + arg.name + '_parameters.pth', map_location=device))

    # Collect data
    student.Q_eval.eval()
    teacher.Q_eval.eval()
    for i in range(arg.episodes):
        observation = env.reset()
        done = False
        with torch.no_grad():
            while not done:
                if env.turn == 1:
                    observation_init = deepcopy(observation)
                    action = student.choose_action(observation)
                    observation_, reward, done, info = env.step(action)
                    observation = observation_
                    student.store_transition(observation_init, action, reward, observation_next, done)
                elif not done:
                    observation_init = deepcopy(observation)
                    action = teacher.choose_action(-observation_init)
                    observation_, reward, done, info = env.step(action)
                    observation = observation_
                    observation_next = deepcopy(observation_)
                    student.store_transition(-observation_init, action, reward, -observation_next, done)
        # Train student network
        student.Q_eval.train()
        loss = student.learn()
        student.Q_eval.eval()
        # Test performance against teacher and make update
        with torch.no_grad():
            if i % 500 == 0 and i > 0:
                eps_student = student.epsilon
                eps_teacher = teacher.epsilon
                student.epsilon = arg.eps_eval
                teacher.epsilon = 0.
                _ = validation(env, student, random_player, 400, i, loss, arg, random_player=True)
                make_new_teacher = validation(env, student, teacher, 400, i, loss, arg)
                student.epsilon = eps_student
                teacher.epsilon = eps_teacher
                if make_new_teacher:
                    torch.save(student.Q_eval.state_dict(), 'parameters/' + arg.name + '_parameters.pth')
                    teacher.Q_eval.load_state_dict(student.Q_eval.state_dict())
                    teacher.Q_eval.eval()
                    teacher.Q_target.load_state_dict(student.Q_target.state_dict())
                    print("***** Upgrade: NEW TEACHER*****")

        if tau:
            for target_param, local_param in zip(student.Q_target.parameters(), student.Q_eval.parameters()):
                target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
                student.Q_target.eval()
        elif i % arg.t_update == 0:
            student.Q_target.load_state_dict(student.Q_eval.state_dict())
            student.Q_target.eval()


if __name__ == '__main__':
    arg = parse_args()
    main(arg)
