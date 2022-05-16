import argparse
from typing import List
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from analyze import aggregate_seed

"""
Example Usage:
    python scripts/plot_scaling.py --csv-path=results/cifar-seq-s12.csv --save-path=figs/cifar-p.png
"""


def get_depth_width_df(df_all: pd.DataFrame, metrics: List[str]):
    n_datas = sorted(df_all['n_data'].unique())
    result = []
    for n_data in n_datas:
        df = df_all[df_all['n_data'] == n_data]
        for h_dim in df_all['h_dim'].unique():
            for n_layers in df_all['n_layers'].unique():
                df_v = df[(df['h_dim'] == h_dim) & (df['n_layers'] == n_layers)]
                result_val = {'n_data': n_data, 'n_layers': n_layers, 'h_dim': h_dim}
                for col in df_v.columns:
                    if col not in metrics:
                        continue
                    score_v = df_v[col].max()
                    result_val[col] = score_v
                result.append(result_val)
    result = pd.DataFrame(result)
    return result


def plot_depth_width_df(df: pd.DataFrame):
    nrows, ncols = 2, 3
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6, 4.5), sharex=True, sharey=True)
    metrics = [c for c in df.columns if '_acc' in c]

    x_axis = np.arange(len(df['n_data'].unique()))
    h_dim_all, n_layers_all = sorted(df['h_dim'].unique()), sorted(df['n_layers'].unique())
    bar_width = 0.25
    for y, n_layers in enumerate(n_layers_all):
        for x, h_dim in enumerate(h_dim_all):
            df_v = df[(df['n_layers'] == n_layers) & (df['h_dim'] == h_dim)]
            for metric, offset in zip(metrics, [-1, 0, 1]):
                label = metric.split('_')[1]  # .split('0')[0]
                ax[y, x].set_title(f'Width {h_dim} Depth {n_layers}', fontsize=10)
                ax[y, x].bar(x_axis+offset*bar_width, df_v[metric], bar_width, label=label)
                if x == 0:
                    ax[y, x].set(ylabel='Recall Accuracy')
                    if y == len(n_layers_all)-1:
                        ax[y, x].set(xlabel='# data')
    fig.tight_layout()
    ax[-1, -1].set_xticks(range(len(df['n_data'].unique())), df['n_data'].unique())
    handles, labels = ax[-1, -1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower right', ncol=len(metrics))
    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv-path', required=True, type=str)
    parser.add_argument('--save-path', required=True, type=str)
    parser.add_argument('--metrics', type=str, nargs='+',
                        default=['test_white0.2_acc', 'test_drop0.25_acc', 'test_mask0.25_acc'])
    args = parser.parse_args()

    df_all = pd.read_csv(args.csv_path).fillna(0)
    df_all = aggregate_seed(df_all=df_all)
    df = get_depth_width_df(df_all=df_all, metrics=args.metrics)
    fig = plot_depth_width_df(df=df)
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    fig.savefig(args.save_path)


if __name__ == "__main__":
    main()
