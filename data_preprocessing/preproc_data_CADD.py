import pandas as pd
import os


file_path_in = "/home/hysong/jsshin/CADD_dataset/CADD"
file_name_in = ["CADD_train.csv", "CADD_dev.csv", "CADD_test.csv"]
file_path_out = "/home/hysong/jsshin/dataset/baseline"
file_name_out = ["train.csv", "valid.csv", "test.csv"]

def target(df):
    # non 0
    # exp 1
    # imp 2
    if df['L.Abusive'] == 0:
        return '0'
    elif df['L.Abusive'] == 1 and df['L.Implicit'] == 0:
        return '1'
    elif df['L.Abusive'] == 1 and df['L.Implicit'] == 1:
        return '2'


if __name__ == "__main__":
    if os.path.exists(file_path_in):
        print("in file path: %s"%file_path_in)
    else:
        print("invalid file path")
        exit()

    for i in range(len(file_name_in)):
        file_n_in = file_name_in[i]
        file_n_out = file_name_out[i]

        file_path = os.path.join(file_path_in, file_n_in)
        if not os.path.exists(file_path):
            print("invalid file path: %s"%file_path)
            exit()
        df = pd.read_csv(file_path, sep=',', encoding='latin_1')
        df['Context'] = df['Title'] + " " + df['Body']
        df['Target'] = df.apply(target, axis=1)
        #print(df.columns)

        # drop if target==nan because of dataset error
        df = df.dropna()

        newdf = df[['Context', 'Comment', 'Target', 'lenComment', 'lenContext']]


        write_path = ""
        if os.path.exists(file_path_out):
            write_path = os.path.join(file_path_out, file_n_out)
            print("out file path: %s\n" % write_path)
        else:
            print("invalid out file path")
            exit()

        newdf.to_csv(write_path, index=False)
        print("Write the file successfully. %s"%file_n_out)


