import torch
import numpy as np
from collections import OrderedDict
from run_paths import RunPaths
from evaluation import evaluation_settings, seed_policy
from evaluation_metrics import EvaluationReport
from model_io import load_weights
from parameters import EVALUATION_PARAMS
from torch.distributions.categorical import Categorical
from utils import convert_to_tensor, save_as_csv


class Evaluator:
    def __init__(self, env, model, run):
        super(Evaluator, self).__init__()
        self.env = env
        self.run = run
        self.local_network = model
        self.model_name = run.algorithm
        self.metrics = EvaluationReport(run, env.net_file_name, env.sumo_steps_green_phase)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.local_network.to(self.device)

        self.episode_step = None
        self.episode_reward = None
        self.episode_action_change = None

        self.rnn_states_actor = None
        self.rnn_states_critic = None


    def load_model(self, model_path, validation=False):
        """
        Load saved trained model
        """
        checkpoint = load_weights(model_path, self.local_network, self.env, self.model_name, self.device)
        if not validation:
            if set(checkpoint['validation_seeds']) & set(EVALUATION_PARAMS.SEEDS):
                raise ValueError('Final evaluation seeds overlap the model selection seeds')
        self.local_network.eval()
        return checkpoint

    def reset(self):
        # Reset evaluator params
        self.episode_step = 0
        self.episode_reward = 0
        self.episode_action_change = 0

        self.rnn_states_actor = None
        self.rnn_states_critic = None

        # Reset Environment params
        self.env.reset_vars()

        # Force each agent to choose the first phase at the initial step
        for tls in self.env.tls_list:
            self.env.tls_dict[tls].set_green_phase(0, self.env.sumo_steps_green_phase)

        obs_n = self.env.observe()

        return obs_n

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
        obs_n = self.reset()
        done = False
        while not done:
            if self.model_name == 'Unicorn' or self.model_name == 'HeteroLight':
                action_dict = OrderedDict()
                phase_vec_dict, phase_mask_dict = self.env.get_phase_vec_mask_dict()
                int_attr_vec_dict = self.env.get_int_attr_vec_dict()

                multi_agent_obs = convert_to_tensor(data=np.array(list(obs_n.values())), data_type=torch.float32, device=self.device)
                multi_agent_phase_vec = convert_to_tensor(data=np.array(list(phase_vec_dict.values())), data_type=torch.float32, device=self.device)
                multi_agent_phase_mask = convert_to_tensor(data=np.array(list(phase_mask_dict.values())), data_type=torch.float32, device=self.device)
                multi_agent_int_attr_vec = convert_to_tensor(data=np.array(list(int_attr_vec_dict.values())), data_type=torch.float32, device=self.device)

                with torch.no_grad():
                    multi_agent_policy, self.rnn_states_actor, _, _, _ = self.local_network(multi_agent_obs,
                                                                                            multi_agent_phase_vec,
                                                                                            multi_agent_int_attr_vec,
                                                                                            multi_agent_phase_mask,
                                                                                            self.rnn_states_actor)
                    multi_agent_policy_dist = Categorical(multi_agent_policy)
                    multi_agent_action = multi_agent_policy_dist.sample().reshape(-1)
                    multi_agent_logprob_p = multi_agent_policy_dist.log_prob(multi_agent_action).reshape(-1)

                action_list = multi_agent_action.detach().cpu().clone().numpy().flatten()
                for i, tls in enumerate(self.env.rl_tls_list):
                    if action_list[i] >= len(self.env.tls_dict[tls].action_space):
                        policy_dist = Categorical(multi_agent_policy[:, i, :].unsqueeze(0))
                        while action_list[i] >= len(self.env.tls_dict[tls].action_space):
                            action = policy_dist.sample().reshape(-1)
                            logprob_p = policy_dist.log_prob(action).reshape(-1)
                            action_list[i] = action.detach().cpu().clone().numpy().flatten()
                            multi_agent_action[i] = action
                            multi_agent_logprob_p[i] = logprob_p

                    action_dict[tls] = action_list[i]

            else:
                raise NotImplementedError

            next_obs_n, r_n, done, info = self.env.step(action_dict)

            obs_n = next_obs_n
            self.episode_step += 1
            self.episode_reward += info[0]
            self.episode_action_change += info[1]

        # collect trip information each run
        self.env.collect_trip_data()
        self.metrics.add_episode(self.env.traffic_data, self.env.trip_data, index, seed)

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
        self.metrics.save()


def main():
    from utils import set_env
    algorithm, checkpoint = evaluation_settings(learning=True)
    seed_policy(EVALUATION_PARAMS.POLICY_SEED)
    run = RunPaths.create('evaluate', algorithm)
    print('Evaluation output:', run.root)
    env = set_env(server_number=0, run=run, test=True, network=EVALUATION_PARAMS.NETWORK)
    if algorithm == 'Unicorn':
        from models.Unicorn import Unicorn as Model
    else:
        from models.HeteroLight import HeteroLight as Model
    model = Model(input_dim=env.tls_obs_space,
                  agent_dim=env.tls_agent_space,
                  int_vec_dim=env.tls_int_attr_space,
                  actor_lr=1e-4, critic_lr=1e-4)
    evaluator = Evaluator(env=env, model=model, run=run)
    saved = evaluator.load_model(checkpoint)
    run.save_config(checkpoint=checkpoint, model_source={
        'training_network': saved['source_network'], 'completed_updates': saved['completed_updates'],
        'schema': saved['schema'], 'selection': saved.get('selection')})
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
