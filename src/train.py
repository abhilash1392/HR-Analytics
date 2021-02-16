# Importing the libraries 
import numpy as np 
import pandas as pd
from sklearn import metrics 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import roc_auc_score 
from sklearn.preprocessing import LabelEncoder 
import argparse
import joblib


# Creating a run function

def run(fold):
    df = pd.read_csv('../input/train_folds.csv')

    # adding a new column 
    df['company_size_new']= "None"

    for i in df['company_size'].values:
        if i in ['<10','10/49','50-99']:
            df['company_size_new'][df['company_size']==i]='small'
        elif i in ['100-500','500-999']:
            df['company_size_new'][df['company_size']==i]='medium'
        elif i in ['1000-4999','5000-9999','10000+']:
            df['company_size_new'][df['company_size']==i]='big'

    # dropping columns 
    df = df.drop(['education_level','experience','company_size'],axis=1)
    
    columns = [f for f in df.columns if f not in ('enrollee_id','target','kfold')]
    numerical_columns = ['city_development_index','training_hours']

    for col in columns:
        if col not in numerical_columns:
            df[col] = df[col].astype(str).fillna("None")
            lbl = LabelEncoder()
            lbl.fit(df[col])
            df[col]= lbl.transform(df[col])

    df_train = df[df['kfold']!=fold].reset_index(drop=True)
    df_valid = df[df['kfold']==fold].reset_index(drop=True)

    X_train = df_train[columns].values
    X_valid = df_valid[columns] .values

    y_train = df_train['target'].values
    y_valid = df_valid['target'].values 

    rf = RandomForestClassifier(criterion='gini',max_depth=7,n_estimators = 400)
    rf.fit(X_train,y_train)
    y_pred= rf.predict_proba(X_valid)[:,1]

    roc_auc = roc_auc_score(y_valid,y_pred)

    print('Fold --> {}   | AUC Score --> {:.3f}'.format(fold,roc_auc))
    joblib.dump(rf,f'../models/dt_{fold}.bin')


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', type = int)
    args = parser.parse_args()
    run(fold = args.fold)

    
