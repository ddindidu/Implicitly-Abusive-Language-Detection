import pandas as pd
import os

file_dir = "/home/hysong/jsshin/dataset/baseline"
file_name = 'test.csv'



if __name__=="__main__":
    file_path = os.path.join(file_dir, file_name)
    if not os.path.exists(file_path):
        raise FileNotFoundError("{} doesn't exist!".format(file_path))

    df = pd.read_csv(file_path)

    exp = df[df['Target']==1]
    imp = df[df['Target']==2]

    exp_rand = exp['Comment'].sample(n=10)
    exp_rand = exp_rand.values.tolist()

    '''
    for i in range(len(exp_rand)):
        print("Exp {}: {}".format(i, exp_rand[i]))
    '''
    
    imp_rand = imp['Comment'].sample(n=20)
    imp_rand = imp_rand.values.tolist()
    for i in range(len(imp_rand)):
        print("Imp {}: {}".format(i, imp_rand[i]))
