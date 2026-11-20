from utils import save_as_csv
import torch
import numpy as np
from collections import OrderedDict
from run_paths import RunPaths
from evaluation import evaluation_settings, seed_policy
from parameters import EVALUATION_PARAMS


class Evaluator:
    def __init__(self, env, run):
        super(Evaluator, self).__init__()
        self.env = env
        self.run = run
        self.agent_name = run.algorithm


        self.episode_step = None
        self.episode_reward = None
        self.episode_action_change = None

    def reset(self):
        # Reset evaluator params
        self.episode_step = 0
        self.episode_reward = 0
        self.episode_action_change = 0

        # Reset Environment params
        self.env.reset_vars()

        return self.env.observe()

    def evaluate(self, index, seed):
        """
        Evaluate the trained model
        @param index: int -> test/evaluation index
        @param seed: seed
        @return:
        """
        # Reset simulations
        self.env.curr_episode = index
        self.env.set_world(seed=seed)
        # Force each agent to choose the first phase at the initial step
        for tls in self.env.tls_list:
            self.env.tls_dict[tls].set_green_phase(0, self.env.sumo_steps_green_phase)
        obs_n = self.reset()
        done = False
        while not done:
            if self.agent_name == 'FixedTime':
                action_dict = OrderedDict()
                for i, tls in enumerate(self.env.tls_list):
                    action_dict[tls] = self.env.tls_dict[tls].get_fixed_time_action()
            elif self.agent_name == 'Greedy':
                action_dict = OrderedDict()
                if self.env.tls_map_dataset == 'resco':
                    for i, tls in enumerate(self.env.tls_list):
                        action_dict[tls] = self.env.tls_dict[tls].get_greedy_action(detector=False)
                else:
                    for i, tls in enumerate(self.env.tls_list):
                        action_dict[tls] = self.env.tls_dict[tls].get_greedy_action(detector=True)

            elif self.agent_name == 'MaxPressure':
                action_dict = OrderedDict()
                if self.env.tls_map_dataset == 'resco':
                    for i, tls in enumerate(self.env.tls_list):
                        action_dict[tls] = self.env.tls_dict[tls].get_pressure_action(detector=False)
                else:
                    for i, tls in enumerate(self.env.tls_list):
                        action_dict[tls] = self.env.tls_dict[tls].get_pressure_action(detector=True)

            else:
                raise NotImplementedError

            next_obs_n, r_n, done, info = self.env.step(action_dict)

            obs_n = next_obs_n
            self.episode_step += 1
            self.episode_reward += info[0]
            self.episode_action_change += info[1]

        # collect trip information each run
        self.env.collect_trip_data()

        print('{} || Episode Length: {} || Episode Reward: {} || Action Change: {}'.format(index,
                                                                                           self.episode_step,
                                                                                           self.episode_reward,
                                                                                           self.episode_action_change / self.episode_step))

    def output_eval_data(self):
        """
        Save the evaluation results (traffic data and trip data)
        @return:
        """
        traffic_data_path = self.run.data / f'{self.env.net_file_name}_traffic.csv'
        trip_data_path = self.run.data / f'{self.env.net_file_name}_trip.csv'
        save_as_csv(file=traffic_data_path, data=self.env.traffic_data)
        save_as_csv(file=trip_data_path, data=self.env.trip_data)


def main():
    from utils import set_env
    algorithm, _ = evaluation_settings(learning=False)
    seed_policy(EVALUATION_PARAMS.POLICY_SEED)
    run = RunPaths.create('evaluate', algorithm)
    print('Evaluation output:', run.root)
    env = set_env(server_number=0, run=run, test=True, network=EVALUATION_PARAMS.NETWORK)
    run.save_config()
    evaluator = Evaluator(env=env, run=run)
    print('Random seeds:', EVALUATION_PARAMS.SEEDS)
    try:
        for i, seed in enumerate(EVALUATION_PARAMS.SEEDS):
            evaluator.evaluate(index=i, seed=seed)
        evaluator.output_eval_data()
    finally:
        if env.simulator.isLoaded():
            env.simulator.close()
    print('Finish output the evaluation data!')


if __name__ == '__main__':
    main()
