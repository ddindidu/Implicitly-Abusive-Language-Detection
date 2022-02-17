import pandas as pd
import os

data_dir_path = './'
file_notabu = 'sentiment_analysis_notabu.csv'
file_exp = 'sentiment_analysis_exp.csv'
file_imp = 'sentiment_analysis_imp.csv'
file_ours_better = 'sentiment_analysis_ours_better.csv'

file_ours_notabu = 'sentiment_analysis_ours_right_notabu.csv'
file_ours_exp = 'sentiment_analysis_ours_right_exp.csv'
file_ours_imp = 'sentiment_analysis_ours_right_imp.csv'


def read_file(dir_path, file_name):
    file_path = os.path.join(dir_path, file_name)
    if os.path.exists(file_path):
        print('File exists: %s'%file_path)
    else:
        raise FileNotFoundError('File is not found: %s'%file_path)

    df = pd.read_csv(file_path)
    return df


def print_result(df_name, num_total, mean_total, std_total, num_pos, mean_pos, std_pos, num_neg, mean_neg, std_neg):
    print("===================Sentiment analysis of {}====================\n".format(df_name))
    print("Total Data")
    print("#: {}".format(num_total))
    print("Confidence Mean: {} ({})".format(mean_total, std_total))
    sent_prob = mean_pos*num_pos - mean_neg*num_neg
    sent_prob = sent_prob/(num_pos + num_neg)
    print("Sentiment probability: {}".format(sent_prob))
    print()

    print("Positive Data")
    print("#: {} ({})".format(num_pos, num_pos/num_total))
    print("Confidence Mean: {} ({})".format(mean_pos, std_pos))
    print()

    print("Negative Data")
    print("#: {} ({})".format(num_neg, num_neg/num_total))
    print("Confidence Mean: {} ({})".format(mean_neg, std_neg))
    print()



def analysis(df, df_name):
    num = len(df)

    # import IPython; IPython.embed();

    mean_pos = df['POSITIVE'].mean()
    std_pos = df['POSITIVE'].std()

    mean_neg = df['NEGATIVE'].mean()
    std_neg = df['NEGATIVE'].std()

    print("============{}==============".format(df_name))
    print("# of data: {}".format(num))
    print("POSITIVE: {} ({})".format(round(mean_pos, 4), round(std_pos, 4)))
    print("NEGATIVE: {} ({})".format(round(mean_neg, 4), round(std_neg, 4)))

    '''
    # average confidence (probability) of total data
    num_total = len(df)
    mean_total = df['probability'].mean()
    std_total = df['probability'].std()

    # Positive data
    num_pos = len(df[df['sentiment'] == 'POSITIVE'])
    mean_pos = df[df['sentiment'] == 'POSITIVE']['probability'].mean()
    std_pos = df[df['sentiment'] == 'POSITIVE']['probability'].std()

    # negative data
    num_neg = len(df[df['sentiment'] == 'NEGATIVE'])
    mean_neg = df[df['sentiment'] == 'NEGATIVE']['probability'].mean()
    std_neg = df[df['sentiment'] == 'NEGATIVE']['probability'].std()

    print_result(df_name, num_total, mean_total, std_total,
                 num_pos, mean_pos, std_pos,
                 num_neg, mean_neg, std_neg)
    '''


if __name__ == '__main__':
    # flag = 'total'
    flag = 'ours'

    if flag == 'total':
        df_notabu = read_file(data_dir_path, file_notabu)
        df_exp = read_file(data_dir_path, file_exp)
        df_imp = read_file(data_dir_path, file_imp)
    else:
        df_notabu = read_file(data_dir_path, file_ours_notabu)
        df_exp = read_file(data_dir_path, file_ours_exp)
        df_imp = read_file(data_dir_path, file_ours_imp)

    df_ours_better = read_file(data_dir_path, file_ours_better)

    # df_imp
    analysis(df_notabu, 'Total Not abusive Data')

    # df_imp
    analysis(df_exp, 'Total Explicit Data')

    # df_imp
    analysis(df_imp, 'Total Implicit Data')

    # df_ours_better
    analysis(df_ours_better, "Implicit Data where only ours performs better")