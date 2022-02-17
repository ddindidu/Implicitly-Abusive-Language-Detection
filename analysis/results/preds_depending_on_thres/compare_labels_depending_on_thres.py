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


def compare_base_ours_0_1(dataset, df):
    thres=80
    colname_thres = "ood_{}".format(thres)

    # if the answer is 0 (Not abu) or 1 (Exp. abu.)
    # also only for 0 1
    df = df[(df['real'] != 2) &
            (df['baseline'] != 2) &
            (df[colname_thres] != 2)]

    df_base_right_thres_wrong = df[(df['baseline'] == df['real'])&
                                   (df[colname_thres] != df['real'])]

    df_base_wrong_thres_right = df[(df['baseline'] != df['real'])&
                                   (df[colname_thres] == df['real'])]

    df_base_right_thres_wrong.to_csv('{}_base_right_ours_wrong_for_0_or_1.csv'.format(dataset['dataname']), index=False)
    df_base_wrong_thres_right.to_csv('{}_base_wrong_ours_right_for_0_or_1.csv'.format(dataset['dataname']), index=False)

    print("File Saving Success")


def compare_base_and_thres(df):
    for thres in threshold:
        colname_thres = "ood_{}".format(thres)

        df_base_right_thres_wrong = df[(df['real'] == df['baseline']) &
                                       (df['real'] != df[colname_thres])]
        df_base_wrong_thres_right = df[(df['real'] != df['baseline']) &
                                       (df['real'] == df[colname_thres])]

        df_base_right_thres_wrong.to_csv('base_right_thres{}_wrong.csv'.format(thres), index=False)
        df_base_wrong_thres_right.to_csv('base_wrong_thres{}_right.csv'.format(thres), index=False)


def compare_base_and_thres_imp(df):
    # only for baseline v.s. threshold 65
    thres = threshold[0]
    colname_thres = "ood_{}".format(thres)
    df_base_right_thres_wrong_imp = df[(df['real'] == df['baseline']) &
                                       (df['real'] != df[colname_thres]) &
                                       (df[colname_thres] == 2)]
    df_base_wrong_thres_right_imp = df[(df['real'] != df['baseline']) &
                                       (df['real'] == df[colname_thres]) &
                                       (df[colname_thres] == 2)]

    df_base_right_thres_wrong_imp.to_csv('base_right_thres{}_wrong_imp.csv'.format(thres), index=False)
    df_base_wrong_thres_right_imp.to_csv('base_wrong_thres{}_right_imp.csv'.format(thres), index=False)

    for thres1, thres2 in zip(threshold1, threshold2):
        colname_thres1 = "ood_{}".format(thres1)
        colname_thres2 = "ood_{}".format(thres2)

        df_base_right_thres_wrong_imp = df[(df['real'] == df['baseline']) &
                                           (df['real'] != df[colname_thres2]) &
                                           (df[colname_thres1] != df[colname_thres2]) &
                                           (df[colname_thres2] == 2)]
        df_base_wrong_thres_right_imp = df[(df['real'] != df['baseline']) &
                                           (df['real'] == df[colname_thres2]) &
                                           (df[colname_thres1] != df[colname_thres2]) &
                                           (df[colname_thres2] == 2)]

        df_base_right_thres_wrong_imp.to_csv('base_right_thres{}_wrong_imp.csv'.format(thres2), index=False)
        df_base_wrong_thres_right_imp.to_csv('base_wrong_thres{}_right_imp.csv'.format(thres2), index=False)


def compare_two_thres(df):
    for thres1, thres2 in zip(threshold1, threshold2):
        colname_th1 = "ood_{}".format(thres1)
        colname_th2 = "ood_{}".format(thres2)

        df_thres1_right_thres2_wrong = df[(df[colname_th1] == df['real']) &
                                          (df[colname_th2]) != df['real']]
        df_thres1_wrong_thres2_right = df[(df[colname_th1] != df['real']) &
                                          (df[colname_th2]) == df['real']]

        df_thres1_right_thres2_wrong.to_csv('thres{}_right_thres{}_wrong.csv'.format(thres1, thres2), index=False)
        df_thres1_wrong_thres2_right.to_csv('thres{}_wrong_thres{}_right.csv'.format(thres1, thres2), index=False)


def compare_two_thres_imp(df):
    for thres1, thres2 in zip(threshold1, threshold2):
        colname_th1 = "ood_{}".format(thres1)
        colname_th2 = "ood_{}".format(thres2)

        df_thres1_right_thres2_wrong_imp = df[(df[colname_th1] == df['real']) &
                                              (df[colname_th2] != df['real']) &
                                              (df[colname_th2] == 2)]
        df_thres1_wrong_thres2_right_imp = df[(df[colname_th1] != df['real']) &
                                              (df[colname_th2] == df['real']) &
                                              (df[colname_th2] == 2)]

        df_thres1_right_thres2_wrong_imp.to_csv('thres{}_right_thres{}_wrong_imp.csv'.format(thres1, thres2), index=False)
        df_thres1_wrong_thres2_right_imp.to_csv('thres{}_wrong_thres{}_right_imp.csv'.format(thres1, thres2), index=False)


if __name__ == '__main__':
    # select the dataset
    dataset = AbuseEval
    # dataset = CADD
    df = read_file('./', dataset['filename'])
    print(df.columns)

    compare_base_ours_0_1(dataset, df)

    # compare_base_and_thres(df)

    # compare_base_and_thres_imp(df)

    # compare_two_thres(df)

    # compare_two_thres_imp(df)
