import pandas as pd
import os
import argparse

CADD = {'data_name': 'CADD',
        'data_dir_path': './../../dataset/CADD/baseline',
        'test_file_name': 'test.csv',
        'label_dir_path': './analysis/results',
        'file_name_label': 'CADD_labels.csv',
        'file_name_baseline': 'CADD_baseline_preds.csv',
        'file_name_ood': 'CADD_ood_preds.csv',
        'file_name_total': 'CADD_total_labels.csv',
        'file_name_ours_better': "CADD_ours_better.csv",
        'file_name_ours_better_base0': "CADD_ours_better_baseline_notabu.csv",
        'file_name_ours_better_base1': "CADD_ours_better_baseline_exp.csv",
}

AbuseEval = {'data_name': 'AbuseEval',
        'data_dir_path': './../../dataset/AbuseEval/baseline',
        'test_file_name': 'test.csv',
        'label_dir_path': './analysis/results',
        'file_name_label': 'AbuseEval_labels.csv',
        'file_name_baseline': 'AbuseEval_baseline_preds.csv',
        'file_name_ood': 'AbuseEval_ood_preds.csv',
        'file_name_total': 'AbuseEval_total_labels.csv',
        'file_name_ours_better': "AbuseEval_ours_better.csv",
        'file_name_ours_better_base0': "AbuseEval_ours_better_baseline_notabu.csv",
        'file_name_ours_better_base1': "AbuseEval_ours_better_baseline_exp.csv",}


def get_args():
    parser = argparse.ArgumentParser()

    # dataset
    parser.add_argument('--data_type', type=str, default='CADD')    # or 'AbuseEval'

    return parser.parse_args()

def read_file(dir_path, file_name):
    file_path = os.path.join(dir_path, file_name)
    if os.path.exists(file_path):
        print('File exists: {}'.format(file_path))
    else:
        raise FileNotFoundError('File is not found: {}'.format(file_path))

    df = pd.read_csv(file_path)
    return df


if __name__=="__main__":
    args = get_args()
    if args.data_type == 'CADD':
        data = CADD
    elif args.data_type == 'AbuseEval':
        data = AbuseEval
    else:
        raise ValueError('Not proper data type!')


    df_data = read_file(data['data_dir_path'], data['test_file_name'])

    df_label = read_file(data['label_dir_path'], data['file_name_label'])
    df_baseline = read_file(data['label_dir_path'], data['file_name_baseline'])
    df_ood = read_file(data['label_dir_path'], data['file_name_ood'])

    assert len(df_data) == len(df_label) == len(df_baseline) == len(df_ood)

    df_label.columns = ['real']
    df_baseline.columns = ['baseline']
    df_ood.columns = ['ood']

    df_total = pd.concat([df_data, df_label, df_baseline, df_ood], axis=1)
    df_total = df_total[['Context', 'Comment', 'Target', 'real', 'baseline', 'ood']]
    df_total.to_csv(data['label_dir_path']+data['file_name_total'], index=False)

    print("df_total len: %d"%len(df_total))

    df_selection = df_total[(df_total['real'] == 2) &
                             (df_total['real'] == df_total['ood']) &
                            (df_total['real'] != df_total['baseline'])]
    print("df_selection len: %d"%len(df_selection))
    # print(df_selection.head(10))

    df_selection.to_csv(os.path.join(data['label_dir_path'], data['file_name_ours_better']), index=True)

    df_selection[df_selection['baseline'] == 0].to_csv(os.path.join(data['label_dir_path'], data['file_name_ours_better_base0']), index=True)
    df_selection[df_selection['baseline'] == 1].to_csv(os.path.join(data['label_dir_path'], data['file_name_ours_better_base1']), index=True)

    print("df_base0 len: %d"%len(df_selection[df_selection['baseline']] == 0))
    print("df_base1 len: %d"%len(df_selection[df_selection['baseline']] == 1))
