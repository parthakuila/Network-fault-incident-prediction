#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 10:34:43 2019

@author: partha
"""

## Load Library
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import preprocessing
import os
#print(os.listdir("../input"))
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn import svm
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import ensemble
from sklearn.metrics import accuracy_score

## Importing data set
%%time
event_type=pd.read_csv("/home/partha/Downloads/telstra-recruiting-network/event_type.csv",error_bad_lines=False)
train = pd.read_csv("/home/partha/Downloads/telstra-recruiting-network/train.csv")
severity_type = pd.read_csv("/home/partha/Downloads/telstra-recruiting-network/severity_type.csv")
log_feature = pd.read_csv("/home/partha/Downloads/telstra-recruiting-network/log_feature.csv")
test = pd.read_csv("/home/partha/Downloads/telstra-recruiting-network/test.csv")
resource_type = pd.read_csv("/home/partha/Downloads/telstra-recruiting-network/resource_type.csv",error_bad_lines=False)
sample_submission = pd.read_csv("/home/partha/Downloads/telstra-recruiting-network/sample_submission.csv")

print("test",test.shape)
print("train",train.shape)

## Input data set head
print('test',test.head())
print('train',train.head(4))
print('sample_submission',sample_submission.head())
print('event_type',event_type.shape,event_type.head(2))
print('severity_type',severity_type.shape,severity_type.head(2))
print('log_feature',log_feature.shape,log_feature.head(2))
print('resource_type',resource_type.shape,resource_type.head(2))

## Fault serverity vizulation
val=list(train['fault_severity'].value_counts())
for i in range(len(val)):
    print(train['fault_severity'].value_counts().index[i],round(val[i]/sum(val)*100),'%')
    
#train['fault_severity'].value_counts(normalize = True)*100

## Data conversation
event_type['id']=pd.to_numeric(event_type['id'],errors='coerce')  #converting object datatype into numeric

## Traing Process
## Merging
def merge_fn(df1,df2,col_name,how_param):
    merged_df=df1.merge(df2,how=how_param,on=col_name)
    return merged_df

train_merge1=merge_fn(train,event_type.drop_duplicates(subset=['id']),'id','left')
train_merge2=merge_fn(train_merge1,severity_type.drop_duplicates(subset=['id']),'id','left')
train_merge3=merge_fn(train_merge2,log_feature.drop_duplicates(subset=['id']),'id','left')
train_merge4=merge_fn(train_merge3,resource_type.drop_duplicates(subset=['id']),'id','left')

train_merge4.shape
train_merge4.head()
train_merge4.columns
list(train_merge4)
## Calculating mean volume
train_merge4['mean_volumn']=train_merge4.groupby(['location','event_type','severity_type','log_feature','resource_type'])['volume'].transform('mean')
train_merge4.head()
train_merge4.dtypes

## checking for missing value
train_merge4.isnull().sum()

## Finding categorical column
cat_col=list(set(train_merge4.columns)-set(train_merge4._get_numeric_data().columns))
cat_col

## Categorical conversation
def categorical_conversion(df,cat_col):
    for i in range(len(cat_col)):
        df[cat_col[i]]=df[cat_col[i]].astype('category') 
    return df

train_merge4=categorical_conversion(train_merge4,cat_col)  
train_merge4.dtypes

## Label encoding
def label_encoding_conversion(df,cat_col):
    le=preprocessing.LabelEncoder()
    for i in range(len(cat_col)):
        df[cat_col[i]]=le.fit_transform(df[cat_col[i]])
    return df

train_merge4.columns
train_merge4=label_encoding_conversion(train_merge4,cat_col)

## Dropping unique values
train_merge4.drop(['id'],axis=1,inplace=True)
target=train_merge4[['fault_severity']]
train_merge4.drop(['fault_severity'],axis=1,inplace=True)
train_merge4.head()
train_merge4.columns
train_merge4.dtypes


## Test data preparetion
test.shape
test.head()
## Test data merging
test_merge1=merge_fn(test,event_type.drop_duplicates(subset=['id']),'id','left')
test_merge2=merge_fn(test_merge1,severity_type.drop_duplicates(subset=['id']),'id','left')
test_merge3=merge_fn(test_merge2,log_feature.drop_duplicates(subset=['id']),'id','left')
test_merge4=merge_fn(test_merge3,resource_type.drop_duplicates(subset=['id']),'id','left')
test_merge4.shape

## Adding new feature mean value
test_merge4['mean_volumn']=test_merge4.groupby(['location','event_type','severity_type','log_feature','resource_type'])['volume'].transform('mean')
severity_type.head()
test_merge4.head(2)

## Categorical column
cat_col
## Categorical conversation
test_merge4=categorical_conversion(test_merge4,cat_col)
test_merge4.dtypes

## Label encoding
test_merge4=label_encoding_conversion(test_merge4,cat_col)
test_merge4.dtypes
## Removing unique column
test_merge4.drop(['id'],axis=1,inplace=True)
train_merge4.columns
test_merge4.columns

##============Build Model ====================
## Logistic Regression
lr=LogisticRegression()
lr.fit(train_merge4,target)
lr_pred=lr.predict(test_merge4)
accuracy_score(pd.DataFrame(lr.predict(train_merge4)),target)

## Random Forest Classifier
rf=RandomForestClassifier()
rf.fit(train_merge4,target)
rf_pred=rf.predict(test_merge4)
accuracy_score(pd.DataFrame(rf.predict(train_merge4)),target)

## GaussianNB
nb=GaussianNB()
nb.fit(train_merge4,target)
nb.predict(test_merge4)
accuracy_score(pd.DataFrame(nb.predict(train_merge4)),target)

## Decision Tree Classifier
dt=tree.DecisionTreeClassifier()
dt.fit(train_merge4,target)
dt.predict(test_merge4)
accuracy_score(pd.DataFrame(dt.predict(train_merge4)),target)

## SVC
svc_ml=svm.SVC()
svc_ml.fit(train_merge4,target)
svc_ml.predict(test_merge4)
accuracy_score(pd.DataFrame(svc_ml.predict(train_merge4)),target)

## AdaBoost Classifier
ada=AdaBoostClassifier()
ada.fit(train_merge4,target)
ada.predict(test_merge4)
accuracy_score(pd.DataFrame(ada.predict(train_merge4)),target)

## KNeighbors Classifier
knn=KNeighborsClassifier()
knn.fit(train_merge4,target)
knn.predict(test_merge4)
accuracy_score(pd.DataFrame(knn.predict(train_merge4)),target)

## Gradient Boosting Classifier
gb=ensemble.GradientBoostingClassifier()
gb.fit(train_merge4,target)
gb_pre=gb.predict(test_merge4)
accuracy_score(pd.DataFrame(gb.predict(train_merge4)),target)

## Model comparison consolidate function
dic_data={}
list1=[]
max_clf_output=[]
tuple_l=()
def data_modeling(X,target,model):
    for i in range(len(model)):
        ml=model[i]
        ml.fit(X,target)
        pred=ml.predict(X)
        acc_score=accuracy_score(pd.DataFrame(ml.predict(X)),target)
        tuple_l=(ml.__class__.__name__,acc_score)
        dic_data[ml.__class__.__name__]=[acc_score,ml]
        list1.append(tuple_l)
        print(dic_data)
    for name,val in dic_data.items():
        if val==max(dic_data.values()):
            max_lis=[name,val]
            print('Maximum classifier',name,val)

    return list1,max_lis

list1,max_lis=data_modeling(train_merge4,target,[AdaBoostClassifier(),KNeighborsClassifier(),
svm.SVC(),RandomForestClassifier(),
tree.DecisionTreeClassifier(),
GaussianNB(),
LogisticRegression(),
ensemble.GradientBoostingClassifier()])
    
model=max_lis[1][1]

## Model score Visualization
modelscore_df=pd.DataFrame(list1,columns=['Classifier',"Accuracy score"])
modelscore_df
modelscore_df['classifier code']=np.arange(8)
modelscore_df
modelscore_df.shape[0]

## Classifier selection
clf_sel=modelscore_df.iloc[modelscore_df['Accuracy score'].idxmax()]
clf_name=clf_sel[0]
modelscore_df.plot.bar(x='classifier code', y='Accuracy score', rot=0)

## Submission file generation
predict_test=rf.predict_proba(test_merge4)
pred_df=pd.DataFrame(predict_test,columns=['predict_0', 'predict_1', 'predict_2'])
submission=pd.concat([test[['id']],pred_df],axis=1)
submission.to_csv('sub.csv',index=False,header=True)

submission
