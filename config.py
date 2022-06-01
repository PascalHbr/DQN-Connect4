import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    # Setup
    parser.add_argument('--gpu', type=int, required=True, help="GPU ID")
    parser.add_argument('--name', required=True)

    # General hyperparameters
    parser.add_argument('--lr', default=1e-4, type=float, help="learning rate of network")
    parser.add_argument('--bn', default=512, type=int, help="batch size")
    parser.add_argument('--episodes', default=1000000, type=float, help="number episodes")

    # Agent hyperparameters
    parser.add_argument('--eps_start', default=1., type=float, help="start value for epsilon")
    parser.add_argument('--eps_end', default=0.05, type=float, help="end value for epsilon")
    parser.add_argument('--eps_decay', default=0.9999, type=float, help="decay value for epsilon")
    parser.add_argument('--eps_teacher', default=0., type=float, help="epsilon value for the teacher")
    parser.add_argument('--eps_eval', default=0.1, type=float, help="student epsilon value during the valuation games")
    parser.add_argument('--gamma', default=0.99, type=float, help="gamma value")
    parser.add_argument('--mem_size', default=1_000_000, type=int, help="memory size")

    # Other hyperparameters
    parser.add_argument('--tau', default=None, type=float, help="tau value (soft update), if not specified hard update is used")
    parser.add_argument('--win', default=0.6, type=float, help="required win percentage for student-teacher-upgrade")
    parser.add_argument('--t_update', default=1000, type=float, help="update routine for target network")

    arg = parser.parse_args()
    return arg