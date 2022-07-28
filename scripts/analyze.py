import argparse
import json
from typing import Any, List
import pandas as pd
import pdb


SWEEPS = ['h-dim', 'act-fn', 'n-layers', 'n-models']


def print_run(df: pd.DataFrame, run_name: str):
    row = df[df['run_name'] == run_name]
    info = {k: v for k, v in row.to_dict().items() if 'train' in k or 'test' in k}
    print(json.dumps(info, indent=4, sort_keys=True))


def aggregate_seed(df_all: pd.DataFrame):
    # Averages the scores across all seeds
    stat_cols = [c for c in df_all.columns.tolist() if 'train' in c or 'test' in c]
    param_cols = [c for c in df_all.columns.tolist()
                  if c not in stat_cols and c not in
                  ['run_name', 'seed', 'path', 'mhn_metric', 'data_start_index', 'forget_every',
                   'run_group', 'run-group', 'acc_thresh']]
    # param_cols = ['n_data', 'h-dim', 'act-fn', 'n-layers', 'n-models']
    # print([c for c in param_cols if len(df_all[c].unique()) > 1])
    grouped = df_all.groupby(param_cols, as_index=False)
    df_all = grouped[stat_cols].mean()
    # df_stdevs = grouped[stat_cols].std()
    # for c in stat_cols:
    #     df_all[f'{c}_std'] = df_stdevs[c]
    return df_all


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
    # pdb.set_trace()
    return result


def assess_metric(df_all: pd.DataFrame, metric: str, ascending: bool = True):
    cols = SWEEPS + [c for c in df_all.columns.tolist() if metric in c]
    df = df_all[cols]
    metric = metric + ('_mse' if ascending else '_acc')
    result = df.sort_values(metric, ascending=ascending)
    print(result[:4])
    # pdb.set_trace()
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

    df_all = pd.read_csv(args.csv_path).fillna(0)
    df_all = apply_filter(df_all=df_all, filters=args.filters)
    df_all = aggregate_seed(df_all=df_all)
    if args.key is not None:
        compare_field_scores(df_all=df_all, key=args.key, ascending=args.ascending)
    elif args.metric is not None:
        assess_metric(df_all=df_all, metric=args.metric, ascending=args.ascending)
    else:
        metrics = ['test_white0.2', 'test_drop0.25', 'test_mask0.25']
        print(f"FILTERS: {args.filters}")
        for metric in metrics:
            print(f"===== Result for {metric} =====")
            assess_metric(df_all=df_all, metric=metric, ascending=args.ascending)
        # raise Exception("One of --key and --metric must be specified.")


if __name__ == "__main__":
    main()
