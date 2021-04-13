import os
import pickle
from collections import OrderedDict
import pandas as pd
import numpy as np
import torch
import seaborn as sns
import matplotlib.pyplot as plt


common_column_names = ['Seed', 'Epoch', 'Algorithm', 'Policy']


class Logger:

    def __init__(self, dir_name, config=None):

        self.dir_name = dir_name
        self.seed_dir = os.path.join(self.dir_name, 'seeds')
        self.model_dir = os.path.join(self.dir_name, 'models')

        self.config = config
        self.epoch_df = self._set_up_results_log()
        self.current_episode_df, self.current_seed, self.current_policy = None, None, None

        self.epoch_returns = []
        self.epoch_jerks = []

        self._save_config()

    def set_up_seed_episode_df(self, policy, seed):
        column_names = common_column_names + ['Return', 'Jerk body', 'Jerk joints', 'Length']
        self.current_policy = policy.get_name()
        self.current_seed = int(seed)
        self.current_episode_df = {name: [] for name in column_names}

    def save_run(self):
        self.current_episode_df['Episode'] = list(range(len(self.current_episode_df['Return'])))
        self.current_episode_df['Mean reward'] = [ret / l for (ret, l) in zip(self.current_episode_df['Return'],
                                                                              self.current_episode_df['Length'])]
        self.current_episode_df = pd.DataFrame.from_dict(self.current_episode_df)
        self.current_episode_df['Smooth return'] = self.current_episode_df['Return'].rolling(self.config.rolling,
                                                                                             min_periods=1).mean()
        self.current_episode_df['Smooth jerk body'] = self.current_episode_df['Jerk body'].rolling(
            self.config.rolling, min_periods=1).mean()
        self.current_episode_df['Smooth jerk joints'] = self.current_episode_df['Jerk joints'].rolling(
            self.config.rolling, min_periods=1).mean()
        self.current_episode_df.to_csv(os.path.join(self.seed_dir,
                                                    self.current_policy + str(self.current_seed) + '.csv'), index=False)
        self.save_experiment()
        if self.config.plot:
            self.plot_episode_df()

    def save_experiment(self):
        df = pd.DataFrame.from_dict(self.epoch_df)
        df.to_csv(os.path.join(self.dir_name, 'epoch_results.csv'), index=False)
        if self.config.plot:
            self.plot_epoch_df()

    def log_episode(self, ret, ep_len, epoch, env):

        try:
            jerk_body, jerk_joints = env.compute_jerk()
        except:
            jerk_body, jerk_joints = 0, 0

        self.current_episode_df['Seed'].append(self.current_seed)
        self.current_episode_df['Epoch'].append(epoch)
        self.current_episode_df['Return'].append(ret)
        self.current_episode_df['Jerk body'].append(jerk_body)
        self.current_episode_df['Jerk joints'].append(jerk_joints)
        self.current_episode_df['Length'].append(ep_len)
        self.current_episode_df['Algorithm'].append(self.config.algorithm)
        self.current_episode_df['Policy'].append(self.current_policy)

        self.epoch_returns.append(ret)
        self.epoch_jerks.append((jerk_body, jerk_joints))

    def log_epoch(self, epoch):

        self.epoch_df['Seed'].append(self.current_seed)
        self.epoch_df['Epoch'].append(epoch)

        mean_ret = np.mean(self.epoch_returns) if len(self.epoch_returns) > 0 else 0.0
        jerks_body, jerks_joints = [j[0] for j in self.epoch_jerks], [j[1] for j in self.epoch_jerks]
        mean_jerk_body = np.mean(jerks_body) if len(jerks_body) > 0 else 0.0
        mean_jerk_joints = np.mean(jerks_joints) if len(jerks_joints) > 0 else 0.0

        self.epoch_df['Mean return'].append(mean_ret)
        self.epoch_df['Mean jerk body'].append(mean_jerk_body)
        self.epoch_df['Mean jerk joints'].append(mean_jerk_joints)
        self.epoch_df['Algorithm'].append(self.config.algorithm)
        self.epoch_df['Policy'].append(self.current_policy)

        self.epoch_returns = []
        self.epoch_jerks = []

    def log(self, message, add_end_line=True):
        if self.config.verbose:
            print(message, flush=True)
        with open(os.path.join(self.dir_name, 'log_file.txt'), 'a+') as file:
            if add_end_line:
                message += '\n'
            file.write(message)

    def save_model(self, model, epoch):
        path = os.path.join(self.model_dir, 'model_' + self.current_policy + '_' + str(epoch) + '.pt')
        if type(model) is tuple:
            torch.save({str(i): m.state_dict() for i, m in enumerate(model)}, path)
        else:
            torch.save(model.state_dict(), path)

    def _set_up_results_log(self):

        os.makedirs(self.seed_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)

        column_names = common_column_names + ['Mean return', 'Mean jerk body', 'Mean jerk joints']
        df = {name: [] for name in column_names}
        return df

    def _save_config(self):

        with open(os.path.join(self.dir_name, 'config.pkl'), 'wb') as file:
            pickle.dump(self.config, file)

        config_dict = self.config.__dict__
        config_dict = OrderedDict(sorted(config_dict.items()))
        with open(os.path.join(self.dir_name, 'config.txt'), 'w') as file:
            for k, v in config_dict.items():
                file.write(str(k) + ': ' + str(v) + '\n')

    def plot_episode_df(self, figsize=(20, 8)):
        ax = sns.lineplot(x='Episode', y='Smooth return', data=self.current_episode_df)
        ax.grid(color='#c7c7c7', linestyle='--', linewidth=1)
        path = os.path.join(self.seed_dir, 'return_' + self.current_policy + str(self.current_seed) + '.pdf')
        plt.savefig(path, bbox_inches='tight')
        plt.close()

        fig, axes = plt.subplots(1, 2, figsize=figsize)

        sns.lineplot(x='Episode', y='Smooth jerk body', data=self.current_episode_df, ax=axes[0])
        axes[0].grid(color='#c7c7c7', linestyle='--', linewidth=1)

        sns.lineplot(x='Episode', y='Smooth jerk joints', data=self.current_episode_df, ax=axes[1])
        axes[1].grid(color='#c7c7c7', linestyle='--', linewidth=1)

        path = os.path.join(self.seed_dir, 'jerk_' + self.current_policy + str(self.current_seed) + '.pdf')
        fig.savefig(path, bbox_inches='tight')
        plt.close()

    def plot_epoch_df(self, figsize=(20, 8)):
        ax = sns.lineplot(x='Epoch', y='Mean return', data=self.epoch_df, ci=self.config.ci, hue='Policy')
        ax.grid(color='#c7c7c7', linestyle='--', linewidth=1)
        path = os.path.join(self.dir_name, 'epoch_returns.pdf')
        plt.savefig(path, bbox_inches='tight')
        plt.close()

        fig, axes = plt.subplots(1, 2, figsize=figsize)

        sns.lineplot(x='Epoch', y='Mean jerk body', data=self.epoch_df, ci=self.config.ci, ax=axes[0], hue='Policy')
        axes[0].grid(color='#c7c7c7', linestyle='--', linewidth=1)

        sns.lineplot(x='Epoch', y='Mean jerk joints', data=self.epoch_df, ci=self.config.ci, ax=axes[1], hue='Policy')
        axes[1].grid(color='#c7c7c7', linestyle='--', linewidth=1)

        path = os.path.join(self.dir_name, 'epoch_jerks.pdf')
        fig.savefig(path, bbox_inches='tight')
        plt.close()
