import argparse
import os
import pandas as pd


"""
1. Collect all 'score.csv' from a particular run group
2. Concatenated them into a single CSV and save
"""


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-group', type=str, default='default')
    return parser


def main():
    args = get_parser().parse_args()
    dfs = []
    group_dir = f'runs/{args.run_group}'
    score_paths = [f'{group_dir}/{name}/score.csv' for name in os.listdir(group_dir)]
    for score_path in score_paths:
        if not os.path.exists(score_path):
            print(f"{score_path} was not found. Continuing...")
            continue
        df = pd.read_csv(score_path)
        dfs.append(df)
    score_df = pd.concat(dfs)
    score_df.to_csv(f'{group_dir}/result.csv', index=False)


if __name__ == "__main__":
    main()
