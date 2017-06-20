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
from sklearn.metrics import log_loss
from itertools import combinations
import numpy as np

cols={1:'smoke',2:'alco',3:'active'}

def get_data():

    data=pd.read_csv('./train_data.csv',sep=',')
    data=data.set_index('id')

    X=data.iloc[:,:-1]
    y=data.iloc[:,-1]

    data_test=pd.read_csv('./test_data.csv',sep=',',index_col='id')


    X_t=data_test
    

    return X,y,X_t

X,y,X_t=get_data()



X['ap_diff']=X['ap_hi']-X['ap_lo']
X_t['ap_diff']=X_t['ap_hi']-X_t['ap_lo']

X['indx_fac']=X['weight']/(X['height']**2)
X_t['indx_fac']=X_t['weight']/(X_t['height']**2)


X_train,X_test,y_train,y_test=train_test_split(X,y)


#XGB

def res_XGB():

    # param_test1 = {
    #     'max_depth':[4,5,6],
    #     'min_child_weight':[1,2]
    # }
    # param_test2={
    # 'gamma':[i/10.0 for i in range(0,5)]}
    
    # param_test4 = {
    #  'subsample':[i/10.0 for i in range(6,10)],
    #  'colsample_bytree':[i/10.0 for i in range(6,10)]
    # }
    
    # param_test5 = {
    #  'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100]
    # }
    
    # param_test6 = {
    #  'n_estimators':[100, 200, 500, 1000]
    # }
    
    
    # estim=XGBClassifier( learning_rate =0.01, n_estimators=500, max_depth=6,min_child_weight=1, gamma=0.2, subsample=0.8, colsample_bytree=0.8,objective= 'binary:logistic', nthread=4, scale_pos_weight=1, seed=27,reg_alpha=1)
    
    # gsearch = GridSearchCV(estim,param_grid = param_test6, scoring='neg_log_loss', cv=3)
    # #clf=estim
    # clf=gsearch
    # # #clf=modelfit(clf,X,y)
    # #
    # clf.fit(X,y)
    # y_pred_xb=clf.predict_proba(X_test)
    # print(log_loss(y_test,y_pred_xb))
    
    
    # # #gsearch1.fit(X,y)
    # print(clf.best_params_, clf.best_score_)
    



    xgbc=XGBClassifier( learning_rate =0.01, n_estimators=500, max_depth=6,min_child_weight=1, gamma=0.2, subsample=0.8, colsample_bytree=0.8,objective= 'binary:logistic', nthread=4, scale_pos_weight=1, seed=27,reg_alpha=1)
    #params={'max_depth':[5,6,7]}
    #rscl=GridSearchCV(xgbc,params,cv=5,scoring='neg_log_loss')
    xgbc.fit(X,y,eval_metric='logloss')
    # # y_pred_xb=rscl.predict_proba(X_test)
    # # print(log_loss(y_test,y_pred_xb))

    y_pred=xgbc.predict_proba(X)
    y_pred_t=xgbc.predict_proba(X_t)

    return y_pred,y_pred_t



#Naive Bayes
from sklearn.naive_bayes import GaussianNB

def res_bayes():
    
    clf=GaussianNB()

    clf.fit(X,y)
    y_pred=clf.predict_proba(X)
    y_pred_t=clf.predict_proba(X_t)
    return y_pred,y_pred_t




#SVC

from sklearn.svm import SVC
def res_svc():
    clf=SVC(C=1.0, kernel='rbf', degree=3, gamma='auto', coef0=0.0, shrinking=True, probability=True, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape=None, random_state=None)
    gscl=clf
    #params1={'C':[0.1,1,10]}#,'gamma':[1e-2,0.1,1]}
    #gscl=GridSearchCV(clf,params1,cv=5,scoring='neg_log_loss')
    #gscl.fit(X_train,y_train)
    #print(gscl.best_params_, gscl.best_score_)
    y_pred=clf.predict_proba(X)
    y_pred_t=clf.predict_proba(X_t)

    return y_pred,y_pred_t



#from sklearn.neural_network import MLPClassifier
#clf=MLPClassifier(hidden_layer_sizes=(300,200, ), activation='relu', solver='adam', alpha=0.0001, batch_size='auto', learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=200, shuffle=True, random_state=None, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

from sklearn.preprocessing import Normalizer

norm=Normalizer()
X=norm.fit_transform(X)
X_t=norm.transform(X_t)
