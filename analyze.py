import argparse
import json
from typing import Any, List
import pandas as pd
import pdb


SWEEPS = ['run_name', 'h-dim', 'act-fn', 'n-layers',
          'sigma-prior', 'n-models']


def print_run(df: pd.DataFrame, run_name: str):
    row = df[df['run_name'] == run_name]
    info = {k: v for k, v in row.to_dict().items() if 'train' in k or 'test' in k}
    print(json.dumps(info, indent=4, sort_keys=True))


def apply_filter(df_all: pd.DataFrame, filters: List[str]):
    if filters is None:
        return df_all
    for ft in filters:
        key, val, vtype = ft.split(':')
        if vtype == 'str':
            pass
        elif vtype == 'int':
            val = int(val)
        elif vtype == 'float':
            val = float(val)
        else:
            raise Exception(f"Invalid filter value {ft}")
        df_all = df_all[df_all[key] == val]
    return df_all


def compare_field_scores(df_all: pd.DataFrame, key: str, ascending: bool = True):
    n_datas = sorted(df_all['n_data'].unique())
    vals = sorted(df_all[key].unique())
    result = []
    for n_data in n_datas:
        df = df_all[df_all['n_data'] == n_data]
        for val in vals:
            df_v = df[df[key] == val]
            result_val = {'n_data': n_data, key: val}
            for col in df_v.columns:
                if ascending:
                    if 'mse' not in col:
                        continue
                    score_v = df_v[col].min()
                else:
                    if 'acc' not in col:
                        continue
                    score_v = df_v[col].max()
                result_val[col] = score_v
            result.append(result_val)
    result = pd.DataFrame(result)
    print(result)
    pdb.set_trace()
    return result


def assess_metric(df_all: pd.DataFrame, metric: str, ascending: bool = True):
    cols = SWEEPS + [c for c in df_all.columns.tolist() if metric in c]
    df = df_all[cols]
    metric = metric + ('_mse' if ascending else '_acc')
    result = df.sort_values(metric, ascending=ascending)
    print(result)
    pdb.set_trace()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv-path', required=True, type=str)
    parser.add_argument('--key', default=None, type=str, help='example: --key h-dim')
    parser.add_argument('--metric', default=None, type=str, help='example: --metric test_white0.2')
    parser.add_argument('--filters', default=None, type=str, nargs="+",
                        help='example: --filters n_data:128:int h-dim:256:int act-fn:relu:str')
    parser.add_argument('--ascending', action='store_true', help='if true use mse, otherwise acc')
    args = parser.parse_args()

    df_all = pd.read_csv(args.csv_path)
    df_all = apply_filter(df_all=df_all, filters=args.filters)
    if args.key is not None:
        compare_field_scores(df_all=df_all, key=args.key, ascending=args.ascending)
    elif args.metric is not None:
        assess_metric(df_all=df_all, metric=args.metric, ascending=args.ascending)
    else:
        raise Exception("One of --key and --metric must be specified.")


if __name__ == "__main__":
    main()
