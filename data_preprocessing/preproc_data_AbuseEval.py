import pandas as pd
import os


file_path_in = "/home/hysong/jsshin/AbuseEval_dataset"
file_name_in_text = ["olid-training-v1.0.tsv", "testset-levela.tsv", "testset-levela.tsv"]
file_name_in_label = ["abuseval_offenseval_train.tsv.txt", "abuseval_offenseval_test.tsv.txt", "abuseval_offenseval_test.tsv.txt"]

file_path_out = "/home/hysong/jsshin/dataset/AbuseEval"
file_name_out = ["train.csv", "valid.csv", "test.csv"]

def target(df):
    # non 0
    # exp 1
    # imp 2
    if df['Target'] == 'NOTABU':
        return '0'
    elif df['Target'] == 'EXP':
        return '1'
    elif df['Target'] == 'IMP':
        return '2'


if __name__ == "__main__":
    if os.path.exists(file_path_in):
        print("in file path: %s"%file_path_in)
    else:
        print("invalid file path")
        exit()

    for i in range(3):
        file_in_text = file_name_in_text[i]
        file_in_label = file_name_in_label[i]
        file_out = file_name_out[i]

        file_path_text = os.path.join(file_path_in, file_in_text)
        file_path_label = os.path.join(file_path_in, file_in_label)

        if not os.path.exists(file_path_text):
            print("invalid file path: %s"%file_path_text); exit()
        elif not os.path.exists(file_path_label):
            print("invalid file path: %s"%file_path_label); exit()

        text_df = pd.read_csv(file_path_text, sep='\t')
        label_df = pd.read_csv(file_path_label, sep='\t')

        #print(text_df.head(5))
        #print(label_df.head(5))
        #print(len(text_df), len(label_df))

        df_join = pd.merge(text_df, label_df, left_on='id', right_on='id', how='inner')
        df_join = df_join[['id', 'tweet', 'abuse']]
        #print(df_join.head(5), len(df_join))

        # 일단 raw data save하기 (abuse column이 NOTABU/EXP/IMP로 되어 있음)
        raw_data_save_name = "abuseval_" + file_name_out[i]
        raw_data_save_path = os.path.join(file_path_in, raw_data_save_name)
        df_join.to_csv(raw_data_save_path, index=False)
        print("save raw data: %s"%raw_data_save_name)


        # column name을 Comment, Target으로 변경
        df_join.columns = ['id', 'Comment', 'Target']

        # NOTABU/EXP/IMP -> 0/1/2 로 변환
        df_join['Target'] = df_join.apply(target, axis=1)

        df_baseline = df_join.copy()
        df_ood = df_join.copy()

        # train dataset
        if i==0:
            # ood data preprocessing
            df_ood = df_ood.loc[df_ood['Target'] != '2']

        file_path_out_baseline = os.path.join(file_path_out, 'baseline')
        file_path_out_ood = os.path.join(file_path_out, 'ood')

        file_name_out_baseline = os.path.join(file_path_out_baseline, file_name_out[i])
        file_name_out_ood = os.path.join(file_path_out_ood, file_name_out[i])

        df_baseline.to_csv(file_name_out_baseline, index=False)
        df_ood.to_csv(file_name_out_ood, index=False)

        print("file save succeed: %s"%file_name_out_baseline)
        print("file save succeed: %s" % file_name_out_ood)
