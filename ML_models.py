# -*- coding: utf-8 -*-
"""
Created on Wed Sep 15 11:25:06 2021

@author: Bohdan Pryplotsky

This code is used to get prediction from 3 ML models (LG, XGB, FNN) on 3 feature sets:
•	Feature set 1: patient characteristics 
•	Feature set 2: radiology reports transformed into 768-dimension features using BERT model 
•	Feature set 3: combines patient characteristics and radiology reports
Before running this code, you should run a script Data_Preparation to get prepared dataset (df.pkl) and a script Embedding to get BERT transformation of radiology reports (df_bert_train.pkl) 
1.First, we define a function perform_metrix. 
For data preparation we:
•	Define sets for dependent and independent variables
•	For categorical data perform one hot transform with dummy columns for missing values
•	For all numeric variables create a dummy indicator if missing values occur
•	Transform missing values into mean
•	Standardization for all variables
•	Calculate feature vectors with mean or sum
•	Train, test split (90,10)
•	Oversample the data to have a balanced train set
For performance metrics we:
•	Calculate PR AUC, ROC AUC, Accuracy, Sensitivity, Specificity, PPV, NPV, F1
•	Define confusion matrix and plot it
•	Save results as a list
2.To select important variables, we calculate correlation index and feature importance
3.Perform 2 ML models with 5-fold CV
•	XGB
•	LG 
4. Perform FNN with 5-fold CV
•	Define 3 classes for networks
•	Transform our datasets
•	Define model
•	Start fine tuning
•	Validate our model
"""

#1.Data preparation
#2.Feature importance
#3.Machine learning models
#4.Feedforward neural network
#5.Plots


import numpy as np
import torch
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import collections
from imblearn.under_sampling  import RandomUnderSampler 
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics  import roc_curve
from sklearn.metrics  import auc
from sklearn.model_selection import StratifiedKFold


####Data preparation
#Define global variables
make_balanced_data=True#make balanced dataset
calculate_corr_plot=True#calculate correlation
calculate_feature_importance=True#calculate feature importance
cwd=r'D:\Bohdan'#Working directory 
conf_mat_var=['PR AUC', 'ROC AUC','Accuacy', 'Sensitivity', 'Specificity','NPV', 'Precision', 'Recall', 'F1']#Performance indicators
#Hyperparameters for FNN
epochs = 8#Number of epochs
device = "cuda" if torch.cuda.is_available() else "cpu"#Device to use GPU if available
list_with_results=[]#list with results of the performed models
#Function to calculate performance metrics
def perform_metrix(y_set,predictions,model):  
    #Calculate PR AUC
    precision, recall, thresholds = precision_recall_curve(y_set, predictions)
    print('{} validation set PR AUC: {:.3f}'.format(model, auc(recall, precision))) 
    table_PR_AUC=auc(recall, precision)
    #Calculate ROC AUC
    fpr, tpr, thresholds = roc_curve(y_set, predictions)
    print('{} validation set ROC AUC: {:.3f}'.format(model, auc(fpr, tpr) )) 
    table_ROC_AUC = auc(fpr, tpr) 
    #Find bast thresholds                                                 
    best_thresh = thresholds[np.argmax(tpr - fpr)]
    print('{} Best Threshold: {:.3f}'.format(model, best_thresh))
    #Define confusion matrix
    y_pred_list2=pd.DataFrame(predictions)
    y_pred_list2["ww"]=y_pred_list2[0].map(lambda x: 1 if x > best_thresh else 0)
    print(confusion_matrix(y_set, y_pred_list2["ww"] ))  
    #Calculate Accuracy, Sensitivity, Specificity, Precision, Recall, F1
    conf_TP=confusion_matrix(y_set, y_pred_list2["ww"] )[0,0]
    conf_TN=confusion_matrix(y_set, y_pred_list2["ww"] )[1,1]
    conf_FP=confusion_matrix(y_set, y_pred_list2["ww"] )[0,1]
    conf_FN=confusion_matrix(y_set, y_pred_list2["ww"] )[1,0]
    conf_TN, conf_FP, conf_FN, conf_TP = confusion_matrix(y_set, y_pred_list2["ww"] ).ravel()
    print('TN, FP, FN, TP')
    print(conf_TN, conf_FP, conf_FN, conf_TP)
    print('{} validation set Accuacy: {:.3f}'.format(model, (conf_TP+conf_TN)/(conf_TP+conf_TN+conf_FP+conf_FN)))
    print('{} validation set Sensitivity: {:.3f}'.format(model, (conf_TP)/(conf_TP+conf_FN)))
    print('{} validation set Specificity: {:.3f}'.format(model, (conf_TN)/(conf_TN+conf_FP)))
    print('{} validation set NPV: {:.3f}'.format(model, (conf_TN)/(conf_TN+conf_FN)))
    print('{} validation set Precision: {:.3f}'.format(model, metrics.precision_score(y_set, y_pred_list2["ww"] )))
    print('{} validation set Recall: {:.3f}'.format(model, metrics.recall_score(y_set, y_pred_list2["ww"] )))
    print('{} validation set F1: {:.3f}'.format(model, metrics.f1_score(y_set, y_pred_list2["ww"] )))    
    table_Accuacy = (conf_TP+conf_TN)/(conf_TP+conf_TN+conf_FP+conf_FN)
    table_Sensitivity = (conf_TP)/(conf_TP+conf_FN)
    table_Specificity = (conf_TN)/(conf_TN+conf_FP)
    table_NPV = (conf_TN)/(conf_TN+conf_FN)
    table_Precision = metrics.precision_score(y_set, y_pred_list2["ww"])
    table_Recall = metrics.recall_score(y_set, y_pred_list2["ww"])
    table_F1 = metrics.f1_score(y_set, y_pred_list2["ww"])   
    #Plot confusion matrix
    ax = sns.heatmap(confusion_matrix(y_set, y_pred_list2["ww"] ), annot=True, cmap='Blues', fmt='g',cbar=False, annot_kws={"fontsize":23})
    ax.set_title('Confusion Matrix for ' + model +' \n',fontsize=18);
    ax.set_xlabel('\nPredicted Values',fontsize=14)
    ax.set_ylabel('Actual Values ',fontsize=14)
    ax.set_yticklabels(['Survivors', 'Non-survivors'], minor=False, va="center",fontsize=14)
    ax.set_xticklabels(['Survivors', 'Non-survivors'], minor=False, va="center",fontsize=14);
    list_with_results.append([table_PR_AUC, table_ROC_AUC, table_Accuacy, table_Sensitivity, table_Specificity,table_NPV, table_Precision, table_Recall, table_F1])
#Function for earlystopping
class EarlyStopping:
        """Early stops the training if validation loss doesn't improve after a given patience."""
        def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
            """
            Args:
                patience (int): How long to wait after last time validation loss improved.
                                Default: 7
                verbose (bool): If True, prints a message for each validation loss improvement. 
                                Default: False
                delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                                Default: 0
                path (str): Path for the checkpoint to be saved to.
                                Default: 'checkpoint.pt'
                trace_func (function): trace print function.
                                Default: print            
            """
            self.patience = patience
            self.verbose = verbose
            self.counter = 0
            self.best_score = None
            self.early_stop = False
            self.val_loss_min = np.Inf
            self.delta = delta
            self.path = path
            self.trace_func = trace_func
        #Compare the loss and calculate the stopping counter, else save function
        def __call__(self, val_loss, model):
            score = -val_loss
            if self.best_score is None:
                self.best_score = score
                self.save_checkpoint(val_loss, model)
            elif score < self.best_score + self.delta:
                self.counter += 1
                self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = score
                self.save_checkpoint(val_loss, model)
                self.counter = 0
        #Function to save the best model
        def save_checkpoint(self, val_loss, model):
            '''Saves model when validation loss decrease.'''
            if self.verbose:
                self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
            torch.save(model.state_dict(), self.path)
            self.val_loss_min = val_loss
#Read dataframe
with open( cwd + '/' +  'df' + '.pkl' ,'rb') as path_name:# load df, 'rb' specifies 'read'
  dataframe = pickle.load(path_name)
#Define sets of variables
X_set=dataframe.copy()
#Columns with categorical variables for woe/one-hot transformation
cols_for_woe=['Group','Bodypart']
#Columns with numerical variables
cols_for_X=['Len','Age','text'] + cols_for_woe
#Columns with numerical variables and variables, which iundicate mising values for numeric varaibles
cols_for_missing=['Len','Age']
indicator_columns=['Fallpseudonym2','PatientenPseudonym','Zeitpunkt_UntersuchungsStart2']
#Define set with dependent variable
y_set=X_set[['Event'] + indicator_columns]   
X_set.loc[X_set['text'].isna()==True, 'text'] ='' 
#Define set with features
X_set=X_set[cols_for_X + indicator_columns]
#One hot transform with dummy col for missing values
one_hot = pd.get_dummies(X_set[cols_for_woe],dummy_na=True)
one_hot=one_hot.loc[:,one_hot.columns.duplicated()==False]
#Drop columns as it is now encoded
X_set = X_set.drop(cols_for_woe,axis = 1)
#Join the encoded df
X_set = X_set.join(one_hot)
X_set=X_set.loc[:,X_set.columns.duplicated()==False]
#For all numeric variables create a dummy indicator if missing values occure
for col in X_set[cols_for_missing]:
    X_set[f"{col}_missing"] = (X_set[col].isnull().astype(int))
#Transfrom missing values into mean
my_imputer = SimpleImputer(strategy ='constant',fill_value=-1)
#Replace missing values
X_set[cols_for_missing] = my_imputer.fit_transform(X_set[cols_for_missing])
#Standartization for X set
sc = StandardScaler()
X_set[X_set.columns.difference(indicator_columns+['text'])] = sc.fit_transform(X_set[X_set.columns.difference(indicator_columns+['text'])])
#Sort dataframe
mean_df=X_set.sort_values(['PatientenPseudonym','Zeitpunkt_UntersuchungsStart2'])
#For each variables callculate feaure vactors with mean or sum
#Mean for categorical variables
X_set_2a = mean_df.groupby('Fallpseudonym2')[one_hot.columns].mean()
#Sum for numerical variables
X_set_2b = mean_df.groupby('Fallpseudonym2')['Len'].sum()
#Mean for numerical variables
X_set_2c = mean_df.groupby('Fallpseudonym2')['Age'].mean()
#Merge those 3 sets
X_set_2=pd.concat([X_set_2a,X_set_2b,X_set_2c],axis=1)
#Define three feature sets. merged_df - df with features, which will be used for modeling
feature_set_1=True
feature_set_2=False
feature_set_3=False
if feature_set_3:
    with open(cwd +"\df_bert_train.pkl", 'rb') as file:  
        df_bert_train = pickle.load(file)  
    merged_df=X_set_2.merge(df_bert_train, left_index=True, right_index=True, how='left')
    merged_df=merged_df.fillna(0.01)
if feature_set_2:
    with open(cwd +"\df_bert_train.pkl", 'rb') as file:  
        df_bert_train = pickle.load(file)  
    X_set_2d = pd.DataFrame(mean_df.Fallpseudonym2.unique(),columns=['Fallpseudonym2'])
    merged_df=X_set_2d.merge(df_bert_train, left_on='Fallpseudonym2', right_index=True, how='left')
    merged_df=merged_df.sort_values(['Fallpseudonym2'])
    merged_df=merged_df.set_index('Fallpseudonym2')
    merged_df=merged_df.fillna(0.01)
if feature_set_1:
    merged_df=X_set_2.copy()
#Createfeaure vactor for dependent variable
#Sort
mean_dfy=y_set.sort_values(['PatientenPseudonym','Zeitpunkt_UntersuchungsStart2'])
#Set index
mean_dfy=mean_dfy.set_index('Fallpseudonym2')
#Callculate feaure vactors for dependent variable with sum
y_set_2=mean_dfy.groupby('Fallpseudonym2')['Event'].max()
#Train, test, validation split (70,15,15)
X_set_train, X_set_test, y_set_train, y_set_test = train_test_split(merged_df, y_set_2, test_size=0.1,random_state=1)
X_set_train=X_set_train.sort_index()
X_set_test=X_set_test.sort_index()
y_set_train=y_set_train.sort_index()
y_set_test=y_set_test.sort_index()
if make_balanced_data:
    #Oversampling
    ros = RandomUnderSampler(random_state=1)
    x_ros, y_ros = ros.fit_resample(X_set_train, y_set_train)
    print('Original dataset shape', collections.Counter(y_set_train))
    print('Resample dataset shape', collections.Counter(y_ros))
    X_set_train=x_ros.copy()
    y_set_train=y_ros.copy()


####Corrplot and feature importance
if calculate_corr_plot:
    #Calculate correlation matrix
    corr= X_set_train.corr()
    f,ax = plt.subplots(figsize=(18, 15))
    sns.heatmap(corr, annot=True, cmap='Reds',annot_kws={"size": 30},cbar=False)
if calculate_feature_importance:
    #Define xgb model
    xgb2 = xgb.XGBClassifier(max_depth=8, n_estimators=300, val_metric="logloss", learning_rate = 0.1, colsample_bytree = 0.4,early_stopping_rounds=10, use_label_encoder=False, random_state= 1)
    xgb2.fit(X_set_train, y_set_train)
    #Calculate feature importance 
    perm_imp = permutation_importance(xgb2, X_set_train, y_set_train.ravel(), scoring='roc_auc', n_repeats=5, random_state=1)    
    indices = np.argsort(perm_imp['importances_mean'])[::-1]
    plt.figure()
    plt.title("Random Forest feature importance via permutation importance")
    plt.bar(range(10),perm_imp['importances_mean'][indices[0:10]])
    plt.xticks(range(10), [X_set_train.columns[i] for i in indices[0:10]] , rotation=90)   
    plt.show()


####Machine learning models
###Logistic regression
lr1=LogisticRegression(penalty='l2',C=100)
cv=StratifiedKFold(n_splits=5)
list_with_results=[]
for i, (tr_idx, vl_idx) in enumerate(cv.split(X_set_train, y_set_train)):
    xtr, xvl = X_set_train.iloc[tr_idx], X_set_train.iloc[vl_idx]
    ytr, yvl = y_set_train.iloc[tr_idx], y_set_train.iloc[vl_idx]
    lr1.fit(xtr, ytr)  
    perform_metrix(yvl,lr1.predict_proba(xvl)[:, 1], "XGB")
table_with_results_lr_f1=pd.DataFrame(list_with_results, columns=conf_mat_var)
table_with_results_lr_f1.mean()
table_with_results_lr_f1.std()
perform_metrix(y_set_test,lr1.predict_proba(X_set_test)[:, 1], "LR")
###XGBoost
#Train XGBoost model with default parameters
xgb2 = xgb.XGBClassifier(max_depth=8, n_estimators=300, val_metric="logloss", learning_rate = 0.1, colsample_bytree = 0.4, early_stopping_rounds=10, use_label_encoder=False, random_state=1)
cv=StratifiedKFold(n_splits=5)
list_with_results=[]
for i, (tr_idx, vl_idx) in enumerate(cv.split(X_set_train, y_set_train)):
    xtr, xvl = X_set_train.iloc[tr_idx], X_set_train.iloc[vl_idx]
    ytr, yvl = y_set_train.iloc[tr_idx], y_set_train.iloc[vl_idx]
    xgb2.fit(xtr, ytr)  
    perform_metrix(yvl,xgb2.predict_proba(xvl)[:, 1], "XGB")
table_with_results_xgb_f1=pd.DataFrame(list_with_results, columns=conf_mat_var)
table_with_results_xgb_f1.mean()
table_with_results_xgb_f1.std()
perform_metrix(y_set_test,xgb2.predict_proba(X_set_test)[:, 1], "XGB")


####Feedforward neural network
##Define 3 classes for networks
#Define train and test data function for indexing the datasets
class trainData(Dataset):   
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data      
    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]        
    def __len__ (self):
        return len(self.X_data)  
class testData(Dataset):
    def __init__(self, X_data):
        self.X_data = X_data    
    def __getitem__(self, index):
        return self.X_data[index]    
    def __len__ (self):
        return len(self.X_data)  
cv=StratifiedKFold(n_splits=5)
list_with_results=[]
for i, (tr_idx, vl_idx) in enumerate(cv.split(X_set_train, y_set_train)):
    early_stopping = EarlyStopping(patience=20)
    early_stopping.best_score=None
    ##Transform our datasets
    #Prepare train, test, validation  sets using function above
    xtr, xvl = X_set_train.iloc[tr_idx], X_set_train.iloc[vl_idx]
    ytr, yvl = y_set_train.iloc[tr_idx], y_set_train.iloc[vl_idx]
    train_data = trainData(torch.FloatTensor(xtr.values), torch.FloatTensor(ytr.values))
    test_data = trainData(torch.FloatTensor(xvl.values), torch.FloatTensor(yvl.values))
    val_data = testData(torch.FloatTensor(xvl.values))
    #Split datasets into batchs
    train_loader = DataLoader(dataset=train_data, batch_size=256, shuffle=True)
    test_loader = DataLoader(dataset=test_data, batch_size=256, shuffle=True)
    val_loader = DataLoader(dataset=val_data, batch_size=1)
    ##Define model
    class binaryClassification(nn.Module):
      def __init__(self, input_shape):
        #Define operator with layers of the model
        super(binaryClassification, self).__init__()
        #Define linear transformation with different number features
        self.fc1 = nn.Linear(input_shape, 25)
        self.fc2 = nn.Linear(25, 50)
        self.fc3 = nn.Linear(50, 10)
        self.fc4 = nn.Linear(10, 1) 
        #Define layers with dropout
        self.dropout = nn.Dropout(p=0.078)
        #Define batch normalization of different size
        self.batchnorm1 = nn.BatchNorm1d(25)
        self.batchnorm2 = nn.BatchNorm1d(50)
        self.batchnorm3 = nn.BatchNorm1d(10)   
      def forward(self, x):
        #Define passing operator. It passes X through the operations defined in the __init__ method
        x = F.relu(self.fc1(x))
        x = self.batchnorm1(x)
        x = F.relu(self.fc2(x))
        x = self.batchnorm2(x)
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.batchnorm3(x)
        x = self.dropout(x)
        x = self.fc4(x)
        return x
    #Define model with input argument
    model = binaryClassification(input_shape=X_set_train.shape[1])
    #Define SGD optimizer 
    optimizer = torch.optim.SGD(model.parameters(),lr=0.107)
    #Model to GPU if avaliable
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    #Define loss function
    criterion = nn.BCEWithLogitsLoss()
    #Start fine tuning
    model.train()
    loss_values = []
    loss_values_val = []
    ##Start fine tuning  
    for e in range(1, epochs+1):
        epoch_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()#Zeros the gradients accumulated from the previous step
            y_pred = model(X_batch)#Make a prediction
            loss = criterion(y_pred, y_batch.unsqueeze(1))#Calculate a loss
            loss.backward()#Performs backpropagation
            optimizer.step()#Updates the weights in our neural network based on the results of backpropagation
            epoch_loss += loss.item()#Add loss value       
        epoch_loss_val = 0
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()#Zeros the gradients accumulated from the previous step
            y_pred = model(X_batch)#Make a prediction
            loss = criterion(y_pred, y_batch.unsqueeze(1))#Calculate a loss
            epoch_loss_val += loss.item()#Add loss value            
        loss_values.append(epoch_loss/len(train_loader))
        loss_values_val.append(epoch_loss_val/len(test_loader))     
        early_stopping(loss_values_val[e-1], model)       
        if early_stopping.early_stop:
            print("Early stopping")
            break
        print(f'Epoch {e+0:03}: | Loss: {epoch_loss/len(train_loader):.5f} ')
        print(f'Epoch {e+0:03}: | Loss_val: {epoch_loss_val/len(test_loader):.5f} ')        
    #Load the last checkpoint with the best model
    model.load_state_dict(torch.load('checkpoint.pt'))
    model.eval()
    y_pred_list = []
    with torch.no_grad():
        for X_batch in val_loader:
            X_batch = X_batch.to(device)
            y_test_pred = model(X_batch)        
            y_test_pred = torch.sigmoid(y_test_pred)
            y_pred_tag = torch.round(y_test_pred)
            y_pred_list.append(y_test_pred)
    y_pred_list = [a.squeeze().tolist() for a in y_pred_list] 
    perform_metrix(yvl,y_pred_list, "FNN")
table_with_results_fnn_f1=pd.DataFrame(list_with_results, columns=conf_mat_var)
table_with_results_fnn_f1.mean()
table_with_results_fnn_f1.std()
#Make predictions for test set
val_data2 = testData(torch.FloatTensor(X_set_test.values))
val_loader2 = DataLoader(dataset=val_data2, batch_size=1)
y_pred_list2 = []
model.eval()
with torch.no_grad():
    for X_batch in val_loader2:
        X_batch = X_batch.to(device)
        y_test_pred = model(X_batch)       
        y_test_pred = torch.sigmoid(y_test_pred)
        y_pred_tag = torch.round(y_test_pred)
        y_pred_list2.append(y_test_pred)
y_pred_list2 = [a.squeeze().tolist() for a in y_pred_list2]
perform_metrix(y_set_test,y_pred_list2, "FNN" )
#Plot loss ans ROC AUC
plt.plot(loss_values)
plt.plot(loss_values_val)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Train Loss')
plt.show()


####Plots
fpr, tpr, thresholds = roc_curve(y_set_test, y_pred_list2)
fp_rate, tp_rate, thresholds = roc_curve(y_set_test,xgb2.predict_proba(X_set_test)[:, 1])
fp_rate_lg, tp_rate_lg, thresholds = roc_curve(y_set_test,lr1.predict_proba(X_set_test)[:, 1])
# plot the roc curve for the model
plt.plot([0,1], [0,1], linestyle='--', label='RP')
plt.plot(fp_rate, tp_rate,linestyle="-." , label='XGB', color ="r")
plt.plot(fpr, tpr,linestyle=":",   label='FNN', color ="k")
plt.plot(fp_rate_lg, tp_rate_lg,  label='LR', color ="g")
fp_rate_lg, tp_rate_lg
# axis labels
plt.xlabel('1 - Specificity')
plt.ylabel('Sensitivity')
plt.legend()
# show the plot
plt.show()