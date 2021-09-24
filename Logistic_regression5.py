# -*- coding: utf-8 -*-
"""
Created on Mon Aug 23 09:52:44 2021

@author: bohdan
"""

#Simple logisitc regression	51,68
#Mixed effect logistic regression 	74,34
#XGBoost Classifier	70,43

#Simple logisitc regression	65,35
#Mixed effect logistic regression 	67,16
#XGBoost Classifier	72,20

import pickle
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import category_encoders as ce
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics  import roc_curve
from sklearn.metrics  import auc
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

list_with_auc=[]

cwd=r'F:\Bohdan'
with open( cwd + '/' +  'df' + '.pkl' ,'rb') as path_name:# load df, 'rb' specifies 'read'
  dataframe = pickle.load(path_name)


dataframe.Event.sum()
data_mixed_effects2=dataframe[0:1000 ].copy()
data_mixed_effects2=dataframe.copy()
data_mixed_effects2.loc[data_mixed_effects2.Len2==0,'Len2']=1
data_mixed_effects2['Len2_Log']=np.log(data_mixed_effects2.Len2)
cols_for_woe=['Massnahme_Pat_zustand','Group','Kostentraegertyp','Pat_Geschlecht','Nationalitaet2','Prioritaet']





#Train test split
x1_train, x1_test, y1_train, y1_test = train_test_split(data_mixed_effects2[["Massnahme_Pat_zustand","Len2","Age2","N_reports","Difference2","Group","N_reports_total","Kostentraegertyp","Pat_Geschlecht","Nationalitaet2","Prioritaet"]],data_mixed_effects2.Event, test_size=0.33, random_state=0)
x1_train2, x1_test2 = train_test_split(data_mixed_effects2[['Massnahme_Pat_zustand',"Len2","Age2","N_reports","Event","Fallpseudonym","Difference2","Group","N_reports_total","Kostentraegertyp","Pat_Geschlecht","Nationalitaet2","Prioritaet"]], test_size=0.33, random_state=0)



ce_target = ce.WOEEncoder(cols = cols_for_woe)
ce_target2= ce_target.fit(x1_train[cols_for_woe], y1_train)

X_set_train_woe=ce_target2.transform(x1_train[cols_for_woe])
X_set_train_woe2=ce_target2.transform(x1_train2[cols_for_woe])
X_set_test_woe=ce_target2.transform(x1_test[cols_for_woe])
X_set_test_woe2=ce_target2.transform(x1_test2[cols_for_woe])


x1_train=pd.concat([X_set_train_woe,x1_train.drop(cols_for_woe,axis=1)],axis=1)
x1_train2=pd.concat([X_set_train_woe,x1_train2.drop(cols_for_woe,axis=1)],axis=1)
x1_test=pd.concat([X_set_test_woe,x1_test.drop(cols_for_woe,axis=1)],axis=1)
x1_test2=pd.concat([X_set_test_woe,x1_test2.drop(cols_for_woe,axis=1)],axis=1)

x1_train[["Len2","Age2","N_reports","Difference2","N_reports_total"]].boxplot()
x1_test[["Len2","Age2","N_reports","Difference2","N_reports_total"]].boxplot()
for x in ["Len2","Age2","N_reports","Difference2","N_reports_total"]:
    q75,q25 = np.percentile(x1_train.loc[:,x],[75,25])
    intr_qr = q75-q25
 
    max1 = q75+(1.5*intr_qr)
    min1 = q25-(1.5*intr_qr)
 
    x1_train.loc[x1_train[x] < min1,x] = min1
    x1_train.loc[x1_train[x] > max1,x] = max1
    x1_train2.loc[x1_train2[x] < min1,x] = min1
    x1_train2.loc[x1_train2[x] > max1,x] = max1
    x1_test.loc[x1_test[x] < min1,x] = min1
    x1_test.loc[x1_test[x] > max1,x] = max1
    x1_test2.loc[x1_test2[x] < min1,x] = min1
    x1_test2.loc[x1_test2[x] > max1,x] = max1
    
    

x1_train[["Len2","Age2","N_reports","Difference2","N_reports_total"]].boxplot()
x1_test[["Len2","Age2","N_reports","Difference2","N_reports_total"]].boxplot()

sc = StandardScaler()
x1_train[x1_train.columns] = sc.fit_transform(x1_train[x1_train.columns])
x1_train2[x1_train2.columns.difference(['Fallpseudonym', 'Event'])] = sc.fit_transform(x1_train2[x1_train2.columns.difference(['Fallpseudonym', 'Event'])] )
x1_test[x1_test.columns] = sc.fit_transform(x1_test[x1_test.columns])
x1_test2[x1_test2.columns.difference(['Fallpseudonym', 'Event'])] = sc.fit_transform(x1_test2[x1_test2.columns.difference(['Fallpseudonym', 'Event'])])


    
##### Logistic regression
model = sm.Logit( y1_train, x1_train.astype(float), ).fit()
print(model.summary())
roc_auc_score(y1_test, model.predict(x1_test.astype(float)))
list_with_auc.append(roc_auc_score(y1_test, model.predict(x1_test.astype(float))))
list_with_auc




##### Mixed effect logistic regression
md = smf.mixedlm("Event ~ Len2 +Age2+Group+Difference2+N_reports+N_reports_total+Kostentraegertyp+Pat_Geschlecht+Nationalitaet2+Prioritaet", x1_train2, groups=x1_train2["Fallpseudonym"]).fit()
print(md.summary())

list_with_auc.append(roc_auc_score(x1_test2["Event"], md.predict(x1_test2)))
list_with_auc

#Assumptions
#1 No Multicollinearity

corr= x1_train.corr()
f,ax = plt.subplots(figsize=(18, 15))
sns.heatmap(corr[(corr >= 0.0001) | (corr <= -0.0001)], annot=True);
#2 No Outliers: remove short report(Len2<10) replace long reports lengt by 3sigm
x1_train2[["Len2","Age2","N_reports_total","Event"]].describe()
sns.boxplot(data= x1_train2[["Len2","Age2","N_reports","Difference2","N_reports_total"]]).set_title("GPA and Rank Box Plot")
#3 Linearity
sns.regplot(x= 'Len2', y= 'Event', data= data_mixed_effects2, logistic= True).set_title("GRE Log Odds Linear Plot")
sns.regplot(x= 'PACS_Bilder', y= 'Event', data= data_mixed_effects2, logistic= True).set_title("GRE Log Odds Linear Plot")
sns.regplot(x= 'Age2', y= 'Event', data= data_mixed_effects2, logistic= True).set_title("GPA Log Odds Linear Plot")
sns.regplot(x= 'N_reports_total', y= 'Event', data= data_mixed_effects2, logistic= True).set_title("GRE Log Odds Linear Plot")
#It may be hard to see, but the data does have somewhat of a curve occurring that resembles the S-shaped curve 
#that is required. If a non-S-shaped line were to be present, sometimes a U-shape will be present, 
#how to handle that data needs to be considered.






# Train model
model = RandomForestClassifier()
model.fit(x1_train,y1_train.ravel())
roc_auc_score(y1_test, model.predict(x1_test))


xgb2 = xgb.XGBClassifier(max_depth=10, n_estimators=100,  eval_metric="logloss", learning_rate = 0.1, colsample_bytree = 0.7, use_label_encoder=False, random_state= 1)
xgb2.fit(x1_train, y1_train.ravel())
fp_rate, tp_rate, _ = roc_curve(y1_test, xgb2.predict_proba(x1_test)[:, 1])
print('XGB test set AUC with optimal meta-parameters: {:.4f}'.format(auc(fp_rate, tp_rate) ))


corr= x1_train.corr()
f,ax = plt.subplots(figsize=(18, 15))
sns.heatmap(corr[(corr >= 0.1) | (corr <= -0.1)], annot=True);

#classifier = xgb
perm_imp = permutation_importance(xgb2, x1_train, y1_train, scoring='roc_auc', n_repeats=5, random_state=1)
sorted_idx = perm_imp.importances_mean.argsort()
fig, ax = plt.subplots(figsize=(12,8))
ax.boxplot(perm_imp.importances[sorted_idx].T,
           vert=False, labels=x1_train.columns[:][sorted_idx])
ax.set_title("Permutation importance (test set) of {}".format(str(xgb2)[0:str(xgb2).find('(')]))
fig.tight_layout()
plt.show()









# Setting up the grid of meta-parameters
xgb_param_grid = {
   'colsample_bytree': np.linspace(0.3, 0.6, 4),  # random subspace
   'n_estimators': [ 200, 300, 600],  # ensemble size or number of gradient steps
   'max_depth': [4, 6, 9, 12],   # max depth of decision trees
   'learning_rate': [0.1, 0.01],  # learning rate
    'early_stopping_rounds': [10]}  # early stopping if no improvement after that many iterations
gs_xgb = GridSearchCV(estimator=xgb.XGBClassifier(), param_grid=xgb_param_grid,
                      scoring='roc_auc', cv=5, verbose=2) # cv - cross validation 
gs_xgb.fit(x1_train, y1_train.ravel())

print("Best CV AUC: %0.4f" % gs_xgb.best_score_)
print("Optimal XGB meta-parameters:")
print(gs_xgb.best_params_)

# Find test set AUC of the best XGB classifier
fp_rate, tp_rate, _ = roc_curve(y1_test, gs_xgb.predict_proba(x1_test)[:, 1])
print('XGB test set AUC with optimal meta-parameters: {:.4f}'.format(auc(fp_rate, tp_rate) ))

list_with_auc.append(auc(fp_rate, tp_rate))
list_with_auc




