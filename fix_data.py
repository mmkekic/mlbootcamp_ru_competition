import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn 
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV
import xgboost as xgb
from xgboost import XGBClassifier

from itertools import combinations
import numpy as np


cols={1:'smoke',2:'alco',3:'active'}

def get_data():

    data=pd.read_csv('./ml5/train.csv',sep=';')
    data=data.set_index('id')

    X=data.iloc[:,:-1]
    y=data.iloc[:,-1]

    data_test=pd.read_csv('./ml5/test.csv',sep=';',index_col='id')
    data_test=data_test.replace('None',np.nan)

    for x in cols:
        data_test[cols[x]]=pd.to_numeric(data_test[cols[x]])


    X_t=data_test
    

    return X,y,X_t

X,y,X_t=get_data()

X_total=X.append(X_t)

def fill_missing_vals(X_total):
    #X_total=X.append(X_t)

    iters=['smoke', 'alco', 'active']
    combs=[map(str,x) for x in combinations(iters,3)]+[map(str,x) for x in combinations(iters,2)]+[map(str,x) for x in combinations(iters,1)]


    fixed_data=X_total.copy()



    to_train=X_total.dropna()
    the_rest= X_total[X_total.isnull().any(axis=1)]



    for i in range(len(combs)):

        to_train_X=to_train.drop(combs[i],axis=1)
        to_train_y=to_train[combs[i]]

        for l in range(to_train_y.shape[1]):

            xgbc=xgb.XGBClassifier()
            params={'n_estimators':[100],'max_depth':[3,5,8],'learning_rate':[0.1]}
            clf=GridSearchCV(xgbc,params,cv=3)
            
            
            
            col=combs[i][l]
            
            print(col,i)
            to_fill_X=the_rest.drop(combs[i],axis=1)
            to_fill_X=to_fill_X.dropna()
            indx=to_fill_X.index
            print(len(indx))
            to_train_X_col=to_train_X#.drop(col,axis=1)
            to_train_y_col=to_train[col]
            clf.fit(to_train_X_col,list(to_train_y_col))
            to_fill_y=clf.predict(to_fill_X)
            fixed_data.loc[indx,col]=to_fill_y

    return fixed_data

def fix_outl(X_tot):

    #X_tot=X.append(X_t)
    X_tot['ap_lo']=X_tot['ap_lo'].apply(lambda x:abs(x))
    X_tot['ap_hi']=X_tot['ap_hi'].apply(lambda x:abs(x))
    
    fif_ind=X_tot[(X_tot['ap_lo']<20) & ((X_tot['ap_lo']>4))].index
    X_tot.loc[fif_ind,'ap_lo']=X_tot.loc[fif_ind,'ap_lo'].apply(lambda x: int(x*10.))
    fif_ind=X_tot[(X_tot['ap_lo']<20) & ((X_tot['ap_lo']>4))].index
    X_tot.loc[fif_ind,'ap_lo']=X_tot.loc[fif_ind,'ap_lo'].apply(lambda x: int(x*10.))

    six_ind=X_tot[(X_tot['ap_hi']<30) & ((X_tot['ap_hi']>6))].index
    X_tot.loc[six_ind,'ap_hi']=X_tot.loc[six_ind,'ap_hi'].apply(lambda x: int(x*10.))
    six_ind=X_tot[(X_tot['ap_hi']<30) & ((X_tot['ap_hi']>6))].index
    X_tot.loc[six_ind,'ap_hi']=X_tot.loc[six_ind,'ap_hi'].apply(lambda x: int(x*10.))
    
    firs_ind=X_tot[ (X_tot['ap_lo']>200)]['ap_lo'].index
    X_tot.loc[firs_ind,'ap_lo']=X_tot.loc[firs_ind,'ap_lo'].apply(lambda x: int(x/10.))
    firs_ind=X_tot[ (X_tot['ap_lo']>200)]['ap_lo'].index
    X_tot.loc[firs_ind,'ap_lo']=X_tot.loc[firs_ind,'ap_lo'].apply(lambda x: int(x/10.))
    firs_ind=X_tot[ (X_tot['ap_lo']>200)]['ap_lo'].index
    X_tot.loc[firs_ind,'ap_lo']=X_tot.loc[firs_ind,'ap_lo'].apply(lambda x: int(x/10.))
    
    sec_ind=X_tot[(X_tot['ap_hi']>1000) ]['ap_hi'].index
    X_tot.loc[sec_ind,'ap_hi']=X_tot.loc[sec_ind,'ap_hi'].apply(lambda x: int(abs(x/10.)))
    sec_ind=X_tot[(X_tot['ap_hi']>=500) ]['ap_hi'].index
    X_tot.loc[sec_ind,'ap_hi']=X_tot.loc[sec_ind,'ap_hi'].apply(lambda x: int(abs(x/10.)))
    sec_ind=X_tot[(X_tot['ap_hi']>=500) ]['ap_hi'].index
    X_tot.loc[sec_ind,'ap_hi']=X_tot.loc[sec_ind,'ap_hi'].apply(lambda x: int(abs(x/10.)))

    
    fourth_ind=X_tot[X_tot['ap_hi']<X_tot['ap_lo']].index
    for row_index, row in X_tot.loc[fourth_ind].iterrows():
        if row['ap_lo'] > row['ap_hi']:
            X_tot.set_value(row_index, 'ap_lo' , row['ap_hi'])
            X_tot.set_value(row_index, 'ap_hi' , row['ap_lo'])

    indx=X_tot[ (X_tot['ap_lo']>400) | (X_tot['ap_hi']>=500) | (X_tot['ap_lo']<20) | (X_tot['ap_hi']<60) |(X_tot['ap_hi']<=X_tot['ap_lo'])].index
    

    print('start_training')
    X_copy=X_tot.copy()
    X_tr_all=X_tot.drop(indx,axis=0)
    y_tr_all=X_tr_all[['ap_lo','ap_hi']]
    X_tr_all.drop(['ap_lo','ap_hi'],axis=1,inplace=True)

    print(len(X_tr_all),len(y_tr_all))
    
    xgbc1=xgb.XGBClassifier(n_estimators=10)
    xgbc2=xgb.XGBClassifier(n_estimators=10)
    
    xgbc1.fit(X_tr_all,list(y_tr_all['ap_lo']))
    xgbc2.fit(X_tr_all,list(y_tr_all['ap_hi']))

    #print('finish_training')

    helpr=(X_tot.loc[indx]).drop(['ap_lo','ap_hi'],axis=1)

    #print(helpr.columns)
    X_tot.loc[indx,'ap_lo']=xgbc1.predict(helpr)
    X_tot.loc[indx,'ap_hi']=xgbc2.predict(helpr)
    

    return X_tot

X_total=fix_outl(X_total)
X_total=fill_missing_vals(X_total)

X=X_total.loc[X.index]
X_t=X_total.loc[X_t.index]


X['cardio']=y

X.to_csv('train_data.csv')
X_t.to_csv('test_data.csv')


