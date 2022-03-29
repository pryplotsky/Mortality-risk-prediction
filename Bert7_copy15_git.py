# -*- coding: utf-8 -*-
"""
Created on Wed Sep 15 11:25:06 2021

@author: bohdan
"""

from transformers import BertTokenizer, BertModel
import numpy as np
import torch
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import collections
from imblearn.over_sampling import RandomOverSampler
from torch import nn
import torch.nn.functional as F
from torchmetrics.functional import auroc
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
from sklearn.inspection import permutation_importance
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics  import roc_curve
from sklearn.metrics  import auc

make_balanced_data=True
only_bert_embedding=False
calculate_corr_plot=True
calculate_feature_importance=True
list_with_results=[]
cwd=r'D:\Bohdan'
#Hyperparameters for FNN
learning_rate = 0.1
epochs = 100
tokenizer = BertTokenizer.from_pretrained('D:\Bohdan\Bert')# Get pretrained word embedding
device = "cuda" if torch.cuda.is_available() else "cpu"
bert_model = BertModel.from_pretrained('bert-base-german-cased', proxies={"http":"http://proxy.charite.de:8080","https": "http://proxy.charite.de:8080"}).to(device)#Get pretrained BERT model
#Now let's use our tokenizer to encode our corpus:
def get_document_embedding(document):
  with torch.no_grad():
    inputs = tokenizer(document, return_tensors="pt", max_length=512, truncation=True)#Tokenize the dataset, truncate when passed `max_length`,and pad with 0's when less than `max_length`,save as pytorch  return_tensors,
    inputs.to(device)#Use GPU 
    outputs = bert_model(**inputs)
    document_embedding = outputs.last_hidden_state[0][0]
    document_embedding = document_embedding.unsqueeze_(0)
    return document_embedding
def get_day_embedding(document_list):
    document_embeddings = get_document_embedding(document_list[0]).unsqueeze_(0)
    for document in document_list[1:]:
      document_embeddings = torch.cat([document_embeddings, get_document_embedding(document).unsqueeze_(0)],0)
    print(document_embeddings.shape)
    day_embedding = torch.mean(document_embeddings, dim=0).to(device)
  #day_embedding = torch.cat([day_embedding, torch.tensor([[up_down]]).to(device)],1)
    print(day_embedding.shape)
    return day_embedding
def embeddings_to_tensor(sequence): 
  # pass in slice from dataset, return tensor of embeddings [SEQ_LEN, EMBEDDING_DIM]
  sequence_tensor = sequence[0:1][0]
  for i in range(1,len(sequence)):
    sequence_tensor = torch.cat([sequence_tensor, sequence[i]],0)
  sequence_tensor.to(device)
  return sequence_tensor
#Define metric
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
    print('{} validation set Precision: {:.3f}'.format(model, metrics.precision_score(y_set, y_pred_list2["ww"] )))
    print('{} validation set Recall: {:.3f}'.format(model, metrics.recall_score(y_set, y_pred_list2["ww"] )))
    print('{} validation set F1: {:.3f}'.format(model, metrics.f1_score(y_set, y_pred_list2["ww"] )))
    
    table_Accuacy = (conf_TP+conf_TN)/(conf_TP+conf_TN+conf_FP+conf_FN)
    table_Sensitivity = (conf_TP)/(conf_TP+conf_FN)
    table_Specificity = (conf_TN)/(conf_TN+conf_FP)
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
    list_with_results.append([table_PR_AUC, table_ROC_AUC, table_Accuacy, table_Sensitivity, table_Specificity, table_Precision, table_Recall, table_F1])
with open( cwd + '/' +  'df' + '.pkl' ,'rb') as path_name:# load df, 'rb' specifies 'read'
  dataframe = pickle.load(path_name)
#Define sets of variables
X_set=dataframe.copy()
cols_for_woe=['Group','Bodypart','Maßnahmestatus','Befundstatus','Massnahme_Pat_zustand','Aufnahmediagnose','Geschlecht','Mitgliedsart','Nationalitaet2','AllgemeinePflegestufe2','SpeziellePflegestufe2','Notaufnahmekennzeichen','Bundesland']
cols_for_X=['Len2','Age2','N_reports2','Difference2','text'] + cols_for_woe
cols_for_missing=['Len2','Age2','N_reports2','Difference2']
indicator_columns=['Fallpseudonym2','PatientenPseudonym','Zeitpunkt_UntersuchungsStart2']
#Define set with dependent variable
y_set=X_set[['Event'] + indicator_columns]
#Numeric variables as string
X_set[['Aufnahmediagnose','AllgemeinePflegestufe2','SpeziellePflegestufe2','Geschlecht']]=X_set[['Aufnahmediagnose','AllgemeinePflegestufe2','SpeziellePflegestufe2','Geschlecht']].astype(str)   
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
mean_df=X_set.sort_values(["PatientenPseudonym","Zeitpunkt_UntersuchungsStart2"])
#For each variables callculate feaure vactors with mean or sum
#Mean for categorical variables
X_set_2a = mean_df.groupby('Fallpseudonym2')[one_hot.columns].mean()
#Sum for numerical variables
X_set_2b = mean_df.groupby('Fallpseudonym2')['Len2','Difference2'].sum()
#Mean for numerical variables
X_set_2c = mean_df.groupby('Fallpseudonym2')["Age2",'N_reports2'].mean()
#Merge those 4 sets
X_set_2=pd.concat([X_set_2a,X_set_2b,X_set_2c],axis=1)
###Apply BERT embedding
feature_set_1=True
feature_set_2=False
feature_set_3=False
if feature_set_1:
    merged_df=X_set_2.copy()
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
    merged_df=merged_df.sort_values(["Fallpseudonym2"])
    merged_df=merged_df.set_index('Fallpseudonym2')
    merged_df=merged_df.fillna(0.01)
#Createfeaure vactor for dependent varuable
#Sort
mean_dfy=y_set.sort_values(["PatientenPseudonym","Zeitpunkt_UntersuchungsStart2"])
#Set index
mean_dfy=mean_dfy.set_index('Fallpseudonym2')
#Callculate feaure vactors for dependent variable with sum
y_set_2=mean_dfy.groupby('Fallpseudonym2')['Event'].max()
#Train, test, validation split (70,15,15)
X_set_train, X_set_test, y_set_train, y_set_test = train_test_split(merged_df, y_set_2, test_size=0.15,random_state=1)
X_set_train, X_set_val, y_set_train, y_set_val = train_test_split(X_set_train, y_set_train, test_size=15/85, random_state=1)
X_set_train=X_set_train.sort_index()
X_set_test=X_set_test.sort_index()
X_set_val=X_set_val.sort_index()
y_set_train=y_set_train.sort_index()
y_set_test=y_set_test.sort_index()
y_set_val=y_set_val.sort_index()
if make_balanced_data:
    #Oversampling
    ros = RandomOverSampler(random_state=42)
    x_ros, y_ros = ros.fit_resample(X_set_train, y_set_train)
    print('Original dataset shape', collections.Counter(y_set_train))
    print('Resample dataset shape', collections.Counter(y_ros))
    X_set_train_imb=x_ros.copy()
    y_set_train_imb=y_ros.copy()


####Corrplot and feature importance
if calculate_corr_plot:
    #Calculate correlation matrix
    corr= X_set_train[['Len2','Age2','N_reports2','Difference2']].corr()
    #Filter correlation matrix
    #corr2=corr[(corr >= 0.5) | (corr <= -0.5)]
    #Plot correlation matrix
    f,ax = plt.subplots(figsize=(18, 15))
    sns.heatmap(corr, annot=True, cmap='Reds',annot_kws={"size": 30},cbar=False)
    manual_labels =['Length of reports','Age','Number of reports','Days between reprots']
    ax.set_xticklabels(manual_labels, va="top", fontsize = 23)
    ax.set_yticklabels( manual_labels,va="center",rotation=75, minor=False,fontsize = 23);
if calculate_feature_importance:
    #Define xgb model
    xgb2 = xgb.XGBClassifier(max_depth=8, n_estimators=300, val_metric="logloss", learning_rate = 0.1, colsample_bytree = 0.4,early_stopping_rounds=10, use_label_encoder=False, random_state= 1)
    xgb2.fit(X_set_train, y_set_train)
    #Calculate feature importance 
    perm_imp = permutation_importance(xgb2, X_set_train, y_set_train.ravel(), scoring='roc_auc', n_repeats=5, random_state=1)    
    indices = np.argsort(perm_imp['importances_mean'])[::-1]
    #indices2=indices[(indices>90)]
    plt.figure()
    plt.title("Random Forest feature importance via permutation importance")
    plt.bar(
        range(10),
        perm_imp['importances_mean'][indices[0:10]]
       # yerr=perm_imp['importances_std'][indices]
    )
    plt.xticks(range(10), [X_set_train.columns[i] for i in indices[0:10]] , rotation=90) 
    manual_labels =['Age','GHC 2','GHC 3','Do not have GHC', 'Body part Thorax','ICDC category 1','Days between reprots','Number of reports','GHC 1','Length of reports']
    plt.xticks(range(10), manual_labels , rotation=90)
    plt.ylim([0, 0.025])   
    plt.show()



####Machine learning models
###XGBoost
#Train XGBoost model with default parameters
xgb2 = xgb.XGBClassifier(max_depth=8, n_estimators=300, val_metric="logloss", learning_rate = 0.1, colsample_bytree = 0.4, early_stopping_rounds=10, use_label_encoder=False, random_state= 1)
xgb2.fit(X_set_train, y_set_train)
#Print results for validation and test sets
perform_metrix(y_set_test,xgb2.predict_proba(X_set_test)[:, 1], "XGB")
perform_metrix(y_set_val,xgb2.predict_proba(X_set_val)[:, 1], "XGB" )
    ###Grid search XGBoost
#Define hyperparameters gor grid search
xgb_param_grid = {
   'colsample_bytree': np.linspace(0.3, 0.6, 4),  # random subspace
   'n_estimators': [ 200, 300,600],  # ensemble size or number of gradient steps
   'max_depth': [4, 8,12],   # max depth of decision trees
   'learning_rate': [0.1, 0.01],  # learning rate
    'early_stopping_rounds': [10]}  # early stopping if no improvement after that many iterations
#Train grid search XGBoost model
gs_xgb = GridSearchCV(xgb.XGBClassifier(), param_grid=xgb_param_grid,scoring='roc_auc', cv=5, verbose=10) # cv - cross validation 
gs_xgb.fit(X_set_train, y_set_train.ravel())
print("Optimal XGB meta-parameters:")
print(gs_xgb.best_params_)     
#Print results for validation and test sets
perform_metrix(y_set_test,gs_xgb.predict_proba(X_set_test)[:, 1], "XGB")
perform_metrix(y_set_val,gs_xgb.predict_proba(X_set_val)[:, 1], "XGB" )
###Logistic regression
grid={"C":np.logspace(-3,3,7), "penalty":["l1","l2"]}# l1 lasso l2 ridge
grid={"C":[100], "penalty":["l2"]}# l1 lasso l2 ridge
#Train grid search XGBoost model
logreg_cv=GridSearchCV(LogisticRegression(),grid,cv=5, verbose=10)
logreg_cv.fit(X_set_train.astype(float), y_set_train)
print("tuned hpyerparameters :(best parameters) ",logreg_cv.best_params_)      
#Print results for validation and test sets
perform_metrix(y_set_test,logreg_cv.predict_proba(X_set_test)[:, 1], "LR")
perform_metrix(y_set_val,logreg_cv.predict_proba(X_set_val)[:, 1], "LR" )

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
#Define class EarlyStopping
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
##Transform our datasets
#Prepare train, test, validation sets using function above
train_data = trainData(torch.FloatTensor(X_set_train.values), torch.FloatTensor(y_set_train.values))
val_data = trainData(torch.FloatTensor(X_set_val.values), torch.FloatTensor(y_set_val.values))
test_data = testData(torch.FloatTensor(X_set_test.values))
#Split datasets into batchs
train_loader = DataLoader(dataset=train_data, batch_size=1528, shuffle=True)
val_loader = DataLoader(dataset=val_data, batch_size=1528, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=1)
#Define validation set for calculating individual performance statistics (with batch size 1)
val_data2 = testData(torch.FloatTensor(X_set_val.values))
val_loader2 = DataLoader(dataset=val_data2, batch_size=1)
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
    self.dropout = nn.Dropout(p=0.1)
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
optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate)
#Model to GPU if avaliable
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
#Define loss function
criterion = nn.BCEWithLogitsLoss()
#Start fine tuning
model.train()
loss_values = []
acc_values = []
loss_values_val = []
acc_values_val = []
##Start fine tuning
early_stopping = EarlyStopping(patience=20, verbose=True)
for e in range(1, epochs+1):
    epoch_loss = 0
    epoch_acc = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()#Zeros the gradients accumulated from the previous step
        y_pred = model(X_batch)#Make a prediction
        loss = criterion(y_pred, y_batch.unsqueeze(1))#Calculate a loss
        acc = auroc(y_pred, y_batch.unsqueeze(1).type(torch.cuda.IntTensor))#Calculate ROC AUC
        loss.backward()#Performs backpropagation
        optimizer.step()#Updates the weights in our neural network based on the results of backpropagation
        epoch_loss += loss.item()#Add loss value
        epoch_acc += acc.item()  #Add ROC AUC          
    epoch_loss_val = 0
    epoch_acc_val = 0
    for X_batch, y_batch in val_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()#Zeros the gradients accumulated from the previous step
        y_pred = model(X_batch)#Make a prediction
        loss = criterion(y_pred, y_batch.unsqueeze(1))#Calculate a loss
        acc = auroc(y_pred, y_batch.unsqueeze(1).type(torch.cuda.IntTensor))#Calculate ROC AUC
        epoch_loss_val += loss.item()#Add loss value
        epoch_acc_val += acc.item()  #Add ROC AUC             
    loss_values.append(epoch_loss/len(train_loader))
    acc_values.append(epoch_acc/len(train_loader)) 
    loss_values_val.append(epoch_loss_val/len(val_loader))
    acc_values_val.append(epoch_acc_val/len(val_loader))      
    early_stopping(loss_values_val[e-1], model)       
    if early_stopping.early_stop:
        print("Early stopping")
        break
    print(f'Epoch {e+0:03}: | Loss: {epoch_loss/len(train_loader):.5f} | Acc: {epoch_acc/len(train_loader):.3f}')
    print(f'Epoch {e+0:03}: | Loss_val: {epoch_loss_val/len(val_loader):.5f} | Acc_val {epoch_acc_val/len(val_loader)  :.3f}')        
# load the last checkpoint with the best model
model.load_state_dict(torch.load('checkpoint.pt'))
#Plot loss ans ROC AUC
plt.plot(loss_values)
plt.plot(loss_values_val)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Train Loss')
plt.show()
plt.plot(acc_values)
plt.plot(acc_values_val)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Train Accuracy')
#Make predictions for validation set
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
#Make predictions for test set
y_pred_list = []
model.eval()
with torch.no_grad():
    for X_batch in test_loader:
        X_batch = X_batch.to(device)
        y_test_pred = model(X_batch)        
        y_test_pred = torch.sigmoid(y_test_pred)
        y_pred_tag = torch.round(y_test_pred)
        y_pred_list.append(y_test_pred)
y_pred_list = [a.squeeze().tolist() for a in y_pred_list]
#Save model as pickle file
with open(cwd +"\Pickle_FNN_Model.pkl", 'wb') as file:  
    pickle.dump(model , file)
with open(cwd +"\Pickle_FNN_Model.pkl", 'rb') as file:  
    Pickled_FNN_Model = pickle.load(file)       
#Print results for validation and test sets
perform_metrix(y_set_val,y_pred_list2, "FNN" )
perform_metrix(y_set_test,y_pred_list, "FNN")
#Plots
fpr, tpr, thresholds = roc_curve(y_set_test, y_pred_list)
fp_rate, tp_rate, thresholds = roc_curve(y_set_test,xgb2.predict_proba(X_set_test)[:, 1])
fp_rate_lg, tp_rate_lg, thresholds = roc_curve(y_set_test,logreg_cv.predict_proba(X_set_test)[:, 1])
# plot the roc curve for the model
plt.plot([0,1], [0,1], linestyle='--', label='Random prediction')
plt.plot(fp_rate, tp_rate,linestyle="-." , label='XGB (ROC AUC = 0.935)', color ="r")
plt.plot(fpr, tpr,linestyle=":",   label='FNN (ROC AUC = 0.929)', color ="k")
plt.plot(fp_rate_lg, tp_rate_lg,  label='LR (ROC AUC = 0.915)', color ="g")
fp_rate_lg, tp_rate_lg
# axis labels
plt.xlabel('1 - Specificity')
plt.ylabel('Sensitivity')
plt.legend()
# show the plot
plt.show()














