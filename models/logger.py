import os
import pickle
from collections import OrderedDict
import pandas as pd
import numpy as np


class Logger:

    def __init__(self, dir_name, config=None):

        self.dir_name = dir_name
        self.seed_dir = os.path.join(self.dir_name, 'seeds')

        self.config = config
        self.epoch_df = self._set_up_results_log()
        self.seed_episode_dfs = {}
        self.current_episode_df, self.current_seed = None, None

        self._save_config()

    def set_up_seed_episode_df(self, seed):
        column_names = ['Seed', 'Epoch', 'Return', 'Length']
        self.current_seed = int(seed)
        self.current_episode_df = {name: [] for name in column_names}

    def save_run(self):
        self.current_episode_df['Episode'] = list(range(len(self.current_episode_df['Return'])))
        self.current_episode_df['Mean reward'] = [ret / l for (ret, l) in zip(self.current_episode_df['Return'],
                                                                              self.current_episode_df['Length'])]
        df = pd.DataFrame.from_dict(self.current_episode_df)
        self.seed_episode_dfs[self.current_seed] = df
        df.to_csv(os.path.join(self.seed_dir, str(self.current_seed) + '.csv'), index=False)
        self.save_experiment()

    def save_experiment(self):
        df = pd.DataFrame.from_dict(self.epoch_df)
        df.to_csv(os.path.join(self.dir_name, 'epoch_results.csv'), index=False)

    def log_episode(self, ret, ep_len, epoch):
        self.current_episode_df['Seed'].append(self.current_seed)
        self.current_episode_df['Epoch'].append(epoch)
        self.current_episode_df['Return'].append(ret)
        self.current_episode_df['Length'].append(ep_len)

    def log_epoch(self, epoch_returns, epoch):
        self.epoch_df['Seed'].append(self.current_seed)
        self.epoch_df['Epoch'].append(epoch)
        mean_ret = np.mean(epoch_returns) if len(epoch_returns) > 0 else 0.0
        self.epoch_df['Mean return'].append(mean_ret)

    def log(self, message, add_end_line=True):
        if self.config.verbose:
            print(message, flush=True)
        with open(os.path.join(self.dir_name, 'log_file.txt'), 'a+') as file:
            if add_end_line:
                message += '\n'
            file.write(message)

    def _set_up_results_log(self):

        os.makedirs(self.seed_dir, exist_ok=True)

        column_names = ['Seed', 'Epoch', 'Mean return']
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
