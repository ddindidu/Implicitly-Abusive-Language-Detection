import pandas as pd
import os
from transformers import pipeline

data_dir_path = './../results/'
file_testset = 'CADD_total_labels.csv'
file_ours_better = 'CADD_ours_better.csv'

# file_testset = 'AbuseEval_total_labels.csv'
# file_ours_better = 'AbuseEval_ours_better_selection.csv'


def read_file(dir_path, file_name):
    file_path = os.path.join(dir_path, file_name)
    if os.path.exists(file_path):
        print('File exists: %s'%file_path)
    else:
        raise FileNotFoundError('File is not found: %s'%file_path)

    df = pd.read_csv(file_path)
    return df


def divide_total_df(df):
    df_notabu = df[df['real'] == 0]
    df_exp = df[df['real'] == 1]
    df_imp = df[df['real'] == 2]

    return df_notabu, df_exp, df_imp


def divide_total_df_for_ours_right(df):
    df = df[df['real'] == df['ood']]

    df_notabu = df[df['real'] == 0]
    df_exp = df[df['real'] == 1]
    df_imp = df[df['real'] == 2]

    return df_notabu, df_exp, df_imp


def sentiment_analysis(df, sent_df, classifier):
    for _, row in df.iterrows():
        text = row['Comment']
        label = row['Target']
        sent_anal_result = classifier(text)
        sentiment = sent_anal_result[0]['label']
        probability = sent_anal_result[0]['score']

        if sentiment == 'POSITIVE':
            new_row = {'Comment': text, 'label': label, 'POSITIVE': probability, 'NEGATIVE': 1-probability}
        elif sentiment == 'NEGATIVE':
            new_row = {'Comment': text, 'label': label, 'POSITIVE': 1-probability, 'NEGATIVE': probability}
        else:
            raise(ValueError("Not proper sentiment: %s"%sentiment))

        sent_df = sent_df.append(new_row, ignore_index=True)

    return sent_df


if __name__ == '__main__':
    df_total = read_file(data_dir_path, file_testset)

    '''
    df_notabu = df_total[df_total['real'] == 0]
    df_exp = df_total[df_total['real'] == 1]
    df_imp = df_total[df_total['real'] == 2]
    # df_ours_better = read_file(data_dir_path, file_ours_better) # only when we match Target==ours==2 & baseline != 2 
    '''

    #flag = 'total'
    flag = 'ours'

    if flag == 'total':
        df_notabu, df_exp, df_imp = divide_total_df(df_total)
    elif flag == 'ours':
        df_notabu, df_exp, df_imp = divide_total_df_for_ours_right(df_total)

    df_sentiment_notabu = pd.DataFrame(columns=["Comment", "label", "POSITIVE", "NEGATIVE"])
    df_sentiment_exp = pd.DataFrame(columns=["Comment", "label", "POSITIVE", "NEGATIVE"])
    df_sentiment_imp = pd.DataFrame(columns=["Comment", "label", "POSITIVE", "NEGATIVE"])
    df_sent_ours_better = pd.DataFrame(columns=["Comment", "label", "POSITIVE", "NEGATIVE"])

    classifier = pipeline(task='sentiment-analysis')

    df_sentiment_notabu = sentiment_analysis(df_notabu, df_sentiment_notabu, classifier)
    df_sentiment_exp = sentiment_analysis(df_exp, df_sentiment_exp, classifier)
    df_sentiment_imp = sentiment_analysis(df_imp, df_sentiment_imp, classifier)
    # df_sent_ours_better = sentiment_analysis(df_ours_better, df_sent_ours_better, classifier)

    if flag == 'total':
        df_sentiment_notabu.to_csv('sentiment_analysis_notabu.csv', index=False)
        df_sentiment_exp.to_csv('sentiment_analysis_exp.csv', index=False)
        df_sentiment_imp.to_csv('sentiment_analysis_imp.csv', index=False)
    elif flag == 'ours':
        df_sentiment_notabu.to_csv('sentiment_analysis_ours_right_notabu.csv', index=False)
        df_sentiment_exp.to_csv('sentiment_analysis_ours_right_exp.csv', index=False)
        df_sentiment_imp.to_csv('sentiment_analysis_ours_right_imp.csv', index=False)

    # df_sent_ours_better.to_csv('sentiment_analysis_ours_better.csv', index=False)