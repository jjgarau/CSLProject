import os
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

font = {'size': 16}
matplotlib.rc('font', **font)


def three_column_figures(dir_path, figsize=(27, 7), ci=95):
    path = os.path.join('results', dir_path, 'epoch_results.csv')
    df = pd.read_csv(path)
    df.loc[df['Mean return'] < 0, 'Mean return'] = 0.0
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    ys = ['Mean return', 'Mean jerk body', 'Mean jerk joints']
    for i, (a, y) in enumerate(zip(axes, ys)):
        leg = False if i > 0 else 'auto'
        sns.lineplot(x='Epoch', y=y, data=df, ax=a, hue='Policy', ci=ci, legend=leg)
        a.grid(color='#c7c7c7', linestyle='--', linewidth=1)
    path = os.path.join('results', dir_path, 'epoch_combined.pdf')
    fig.savefig(path, bbox_inches='tight')
    plt.close()


def compute_table_values(dir_path):
    path = os.path.join('results', dir_path, 'epoch_results.csv')
    df = pd.read_csv(path)
    df.loc[df['Mean return'] < 0, 'Mean return'] = 0.0
    last_epoch = df['Epoch'].max()
    for p in ['Baseline', 'Moving average', 'Previous action', 'Action difference', 'Recurrent']:
        for r in ['Mean return', 'Mean jerk body', 'Mean jerk joints']:
            val = df[(df['Policy'] == p) & (df['Epoch'] == last_epoch)][r].mean()
            print(f'For policy {p}, {r} is {val}')


def scatter_plot(dir_path, dir_path_p, figsize=(18, 7), mean=False):
    path = os.path.join('results', dir_path, 'epoch_results.csv')
    path_p = os.path.join('results', dir_path_p, 'epoch_results.csv')
    df = pd.read_csv(path)
    df_p = pd.read_csv(path_p)
    df['Reward shaping'] = 'Without'
    df_p['Reward shaping'] = 'With'
    df = pd.concat([df, df_p], ignore_index=True)
    df.loc[df['Mean return'] < 0, 'Mean return'] = 0.0
    last_epoch = df['Epoch'].max()
    df = df.loc[df['Epoch'] == last_epoch]
    if mean:
        df = df.groupby(['Policy', 'Reward shaping'])[['Mean return', 'Mean jerk body', 'Mean jerk joints']].mean()
        path = os.path.join('results', dir_path, 'scatter_mean.pdf')
    else:
        path = os.path.join('results', dir_path, 'scatter.pdf')
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    ys = ['Mean jerk body', 'Mean jerk joints']
    for i, (a, y) in enumerate(zip(axes, ys)):
        leg = False if i > 0 else 'auto'
        sns.scatterplot(x='Mean return', y=y, data=df, ax=a, hue='Policy', style='Reward shaping', legend=leg, s=75)
        a.grid(color='#c7c7c7', linestyle='--', linewidth=1)
    fig.savefig(path, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    dir_name = 'JerkHumanoid'
    scatter_plot(dir_name, dir_name + '_p')
    scatter_plot(dir_name, dir_name + '_p', mean=True)
