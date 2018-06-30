# handson_dqn.py
import argparse
import os

import chainer
import chainerrl
import gym
from gym.wrappers import Monitor
import numpy as np

from util.qfunctions import QFunction3LP as QFunction
# from util.qfunctions import MyQFunction as QFunction

from util.explorers import ConstantEpsilonGreedy, LinearDecayEpsilonGreedy

from util.replay_buffer import SimpleReplayBuffer, BestReplayBuffer


def get_env(args, is_train=True):
    # env = gym.make(args.env)
    env = gym.make('CartPole-v0')
    # 最大ステップ数を実質取り除く
    env._max_episode_steps = 10 ** 100
    env._max_episode_seconds = 10 ** 100

    if is_train:
        return env  # 訓練時は動画を出力しない

    monitor_dir = os.path.join(args.outdir, "Monitor")
    return Monitor(
        env=env,
        directory=monitor_dir,
        video_callable=(lambda _ep: True),
        force=True
    )


def get_agent(env, args):
    observation_space = env.observation_space
    action_space = env.action_space
    # assert isinstance(action_space, gym.spaces.Discrete)

    # Q-Function
    obs_size = observation_space.shape[0]
    n_actions = action_space.n
    q_func = QFunction(obs_size, n_actions)

    # Optimizer (Adam)
    optimizer = chainer.optimizers.Adam(eps=1e-3)
    optimizer.setup(q_func)

    # 割引率
    gamma = args.gamma

    # Use epsilon-greedy for exploration
    # explorer = ConstantEpsilonGreedy(
    #     epsilon=args.start_epsilon,
    #     random_action_func=env.action_space.sample)
    explorer = LinearDecayEpsilonGreedy(
        args.start_epsilon, args.end_epsilon, args.final_exploration_steps,
        random_action_func=action_space.sample)

    # Experience Replayの設定
    replay_buffer = SimpleReplayBuffer(capacity=args.replay_buffer)
    # replay_buffer = BestReplayBuffer(capacity=args.replay_buffer)

    # gymがnp.float64で出力するが、
    # chainerはnp.float32しか入力できない。
    # そのための変換。
    def phi(x):
        return x.astype(np.float32, copy=False)

    # Agent 生成
    agent = chainerrl.agents.DQN(
        q_func, optimizer, replay_buffer, gamma, explorer,
        replay_start_size=args.replay_start_size,
        target_update_interval=args.target_update_interval,
        update_interval=args.update_interval,
        phi=phi)
    return agent


def train(agent, env, args):
    # train
    n_episodes = args.train_episodes  # 訓練の回数
    max_episode_len = args.max_episode_len  # このステップ数で一旦終了させる
    for i in range(1, n_episodes + 1):
        obs = env.reset()
        reward = 0
        done = False
        sum_of_rewards = 0
        for _ in range(max_episode_len):
            env.render()
            action = agent.act_and_train(obs, reward)  # 訓練
            obs, reward, done, _ = env.step(action)
            sum_of_rewards += reward
            if done:
                break
        if i % 10 == 0:
            print(
                'episode:', i,
                'Rewards:', sum_of_rewards,
                'statistics:', agent.get_statistics()
            )
        agent.stop_episode_and_train(obs, reward, done)


def test(agent, env, args):
    for i in range(args.test_episodes):
        obs = env.reset()
        sum_of_rewards = 0
        while True:
            env.render()
            action = agent.act(obs)
            obs, reward, done, _ = env.step(action)
            sum_of_rewards += reward
            if done:
                break
        print('test episode:', i + 1, 'Rewards:', sum_of_rewards)
        agent.stop_episode()


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--outdir', type=str, default='_output',
                        help='出力ファイル保存先ディレクトリ。'
                             ' 存在しなければ自動生成されます。')
    # parser.add_argument('--env', type=str, default='CartPole-v0',
    #                     help='環境')
    parser.add_argument('--gamma', type=float, default=0.95,
                        help='割引率')
    parser.add_argument('--final-exploration-steps', type=int, default=10 ** 4,
                        help='探索ステップ数')
    parser.add_argument('--start-epsilon', type=float, default=1.0,
                        help='ε-greedy法 の開始ε値')
    parser.add_argument('--end-epsilon', type=float, default=0.1,
                        help='ε-greedy法 の終了ε値')
    parser.add_argument('--replay-buffer', type=int, default=10 ** 6,
                        help='Experience Replay のバッファサイズ')
    parser.add_argument('--replay-start-size', type=int, default=500,
                        help='replay-start-size')
    parser.add_argument('--target-update-interval', type=int, default=100,
                        help='target-update-interval')
    parser.add_argument('--update-interval', type=int, default=1,
                        help='update-interval')
    parser.add_argument('--train-episodes', type=int, default=200,
                        help='訓練エピソード数')
    parser.add_argument('--max-episode-len', type=int, default=2000,
                        help='1回のエピソードの最大ステップ数')
    parser.add_argument('--test-episodes', type=int, default=10,
                        help='検証エピソード数')
    args = parser.parse_args()

    env = get_env(args)
    agent = get_agent(env, args)
    train(agent, env, args)   # 訓練
    agent.save(os.path.join(args.outdir, 'Agent'))  # Agent の保存
    env.close()

    env_test = get_env(args, is_train=False)
    test(agent, env_test, args)   # 訓練した結果でagentを動かす
    env_test.close()
    env_test.env.close()


if __name__ == '__main__':
    main()
