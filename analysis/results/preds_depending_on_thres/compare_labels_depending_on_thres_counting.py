import pandas as pd
import os

AbuseEval = {'dataname': 'AbuseEval',
             'filename': 'AbuseEval_total_labels_and_preds_thres.csv',
             }

CADD = {'dataname': 'CADD',
        'filename': 'CADD_total_labels_and_preds_thres.csv',
        }

threshold = [65, 70, 75, 80]
threshold1 = [65, 70, 75]
threshold2 = [70, 75, 80]


def read_file(dir_path, file_name):
    file_path = os.path.join(dir_path, file_name)
    if os.path.exists(file_path):
        print("File Exists: {}".format(file_path))
    else:
        print("File Not Founded: {}".format(file_path))

    return pd.read_csv(file_path)


def compare_and_count(dataset, df):
    for thres in threshold:
        colname_thres = "ood_{}".format(thres)

        df_base_right_ours_right = df[(df['baseline'] == df['real']) &
                                      (df[colname_thres] == df['real'])]
        df_base_right_ours_wrong = df[(df['baseline'] == df['real']) &
                                      (df[colname_thres] != df['real'])]
        df_base_wrong_ours_right = df[(df['baseline'] != df['real']) &
                                      (df[colname_thres] == df['real'])]
        df_base_wrong_ours_wrong = df[(df['baseline'] != df['real']) &
                                      (df[colname_thres]) != df['real']]

        print("=== Data: {} Threshold: {} ===".format(dataset['dataname'], thres))
        print("base_right_ours_right: {}".format(len(df_base_right_ours_right)))
        print("base_right_ours_wrong: {}".format(len(df_base_right_ours_wrong)))
        print("base_wrong_ours_right: {}".format(len(df_base_wrong_ours_right)))
        print("base_wrong_ours_wrong: {}".format(len(df_base_wrong_ours_wrong)))
        print()


def count_ours_right_answer(dataset, df):
    df_target_2 = df[df['real'] == 2]

    for thres in threshold:
        colname_thres = "ood_{}".format(thres)

        df_base_0 = df_target_2[df_target_2['baseline'] == 0]
        df_base_0_ours_2 = df_base_0[df_base_0[colname_thres] == 2]

        df_base_1 = df_target_2[df_target_2['baseline'] == 1]
        df_base_1_ours_2 = df_base_1[df_base_1[colname_thres] == 2]

        print("===Dataset: {} Thres: {}===".format(dataset['dataname'], thres))
        print("Total Imp: {}".format(len(df_target_2)))
        print("base_0: {} --- ours_correction_2: {}".format(len(df_base_0), len(df_base_0_ours_2)))
        print("base_1: {} --- ours_correction_2: {}".format(len(df_base_1), len(df_base_1_ours_2)))


def count_ours_wrong_answer(dataset, df):
    df_target_0 = df[df['real'] == 0]
    df_target_1 = df[df['real'] == 1]

    for thres in threshold:
        colname_thres = "ood_{}".format(thres)

        df_base_right_0 = df_target_0[df_target_0['baseline'] == df_target_0['real']]
        df_base_right_0_ours_wrong_2 = df_target_0[(df_target_0['baseline'] == df_target_0['real']) &
                                                   (df_target_0[colname_thres] == 2) &
                                                   (df_target_0[colname_thres] != df_target_0['real'])]

        df_base_right_1 = df_target_1[df_target_1['baseline'] == df_target_1['real']]
        df_base_right_1_ours_wrong_2 = df_target_1[(df_target_1['baseline'] == df_target_1['real']) &
                                                   (df_target_1[colname_thres] == 2) &
                                                   (df_target_1[colname_thres] != df_target_1['real'])]

        print("=== Data: {} Threshold: {} ===".format(dataset['dataname'], thres))
        print("Target: 0 (Not abusive)======")
        print("df_target: {}".format(len(df_target_0)))
        print("df_base_right: {}".format(len(df_base_right_0)))
        print("df_base_right_ours_imp: {}".format(len(df_base_right_0_ours_wrong_2)))
        print("Target: 1 (Exp)=====")
        print("df_target: {}".format(len(df_target_1)))
        print("df_base_right: {}".format(len(df_base_right_1)))
        print("df_base_right_ours_imp: {}".format(len(df_base_right_1_ours_wrong_2)))
        print()


if __name__ == '__main__':
    # select the dataset
    # dataset = AbuseEval
    dataset = AbuseEval
    df = read_file('./', dataset['filename'])
    print(df.columns)

    # compare_and_count(dataset, df)
    count_ours_right_answer(dataset,df)
    # count_ours_wrong_answer(dataset, df)