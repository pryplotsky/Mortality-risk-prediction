# -*- coding: utf-8 -*-
"""
Created on Wed Sep 15 11:25:06 2021

@author: bohdan
"""


import pickle
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch import nn, optim
import torch.nn.functional as F
from torchmetrics.functional import auroc
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

import category_encoders as ce

cwd=r'F:\Bohdan'
with open( cwd + '/' +  'df' + '.pkl' ,'rb') as path_name:# load df, 'rb' specifies 'read'
  dataframe = pickle.load(path_name)

X_set=dataframe[0:10000 ].copy()
X_set=dataframe.copy()
X_set.loc[X_set.Len2==0,'Len2']=1
X_set['Len2_Log']=np.log(X_set.Len2)
y_set=X_set.Event
X_set=X_set[["Len2","Age2","N_reports","Difference2","Group","N_reports_total","Kostentraegertyp","Pat_Geschlecht","Nationalitaet2","Prioritaet"]]
cols_for_woe=['Group','Kostentraegertyp','Pat_Geschlecht','Nationalitaet2','Prioritaet']

#for i in cols_for_woe:
#    X_set=pd.concat([X_set,pd.get_dummies(X_set[i], prefix=i)],axis=1).drop([i],axis=1)

X_set[["Len2","Age2","N_reports","Difference2","N_reports_total"]].boxplot()

for x in ["Len2","Age2","N_reports","Difference2","N_reports_total"]:
    q75,q25 = np.percentile(X_set.loc[:,x],[75,25])
    intr_qr = q75-q25
 
    max1 = q75+(1.5*intr_qr)
    min1 = q25-(1.5*intr_qr)
 
    X_set.loc[X_set[x] < min1,x] = min1
    X_set.loc[X_set[x] > max1,x] = max1

X_set[["Len2","Age2","N_reports","Difference2","N_reports_total"]].boxplot()
#X_set_train2, X_set_test2, y_set_train2, y_set_test2 = train_test_split(X_set, y_set, test_size=0.33, random_state=1)
X_set_train, X_set_test, y_set_train, y_set_test = train_test_split(X_set, y_set, test_size=0.33, random_state=1)


ce_target = ce.WOEEncoder(cols = cols_for_woe)
ce_target2= ce_target.fit(X_set_train[cols_for_woe], y_set_train)

X_set_train_woe=ce_target2.transform(X_set_train[cols_for_woe])
X_set_test_woe=ce_target2.transform(X_set_test[cols_for_woe])


X_set_train=pd.concat([X_set_train_woe,X_set_train.drop(cols_for_woe,axis=1)],axis=1)
X_set_test=pd.concat([X_set_test_woe,X_set_test.drop(cols_for_woe,axis=1)],axis=1)


X_set_train[["Len2","Age2","N_reports","Difference2","N_reports_total"]].boxplot()
for x in ["Len2","Age2","N_reports","Difference2","N_reports_total"]:
    q75,q25 = np.percentile(X_set_train.loc[:,x],[75,25])
    intr_qr = q75-q25
 
    max1 = q75+(1.5*intr_qr)
    min1 = q25-(1.5*intr_qr)
 
    X_set_train.loc[X_set_train[x] < min1,x] = min1
    X_set_train.loc[X_set_train[x] > max1,x] = max1
    X_set_test.loc[X_set_test[x] < min1,x] = min1
    X_set_test.loc[X_set_test[x] > max1,x] = max1
X_set_train[["Len2","Age2","N_reports","Difference2","N_reports_total"]].boxplot()
X_set_test[["Len2","Age2","N_reports","Difference2","N_reports_total"]].boxplot()

sc = StandardScaler()
X_set_train[X_set_train.columns] = sc.fit_transform(X_set_train[X_set_train.columns])
X_set_test[X_set_test.columns] = sc.fit_transform(X_set_test[X_set_test.columns])
y_set_test
## train data
class trainData(Dataset):
    
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data
        
    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]
        
    def __len__ (self):
        return len(self.X_data)


train_data = trainData(torch.FloatTensor(X_set_train.values), torch.FloatTensor(y_set_train.values))

## test data    
class testData(Dataset):
    
    def __init__(self, X_data):
        self.X_data = X_data
        
    def __getitem__(self, index):
        return self.X_data[index]
        
    def __len__ (self):
        return len(self.X_data)
    

test_data = testData(torch.FloatTensor(X_set_test.values))

#trainset = dataset(torch.FloatTensor(X_set.values), torch.FloatTensor(y_set.values))
#trainset = dataset(torch.FloatTensor(X_set.values), torch.FloatTensor(y_set.values))
#trainloader = DataLoader(trainset,batch_size=644,shuffle=False)
#testloader = DataLoader(dataset=test_data, batch_size=1)
train_loader = DataLoader(dataset=train_data, batch_size=1028)
test_loader = DataLoader(dataset=test_data, batch_size=1)

#Train test split
#x1_train, x1_test, y1_train, y1_test = train_test_split(data_mixed_effects2[["Len2","Age2","N_reports","Difference2","Group","N_reports_total","Kostentraegertyp","Pat_Geschlecht","Nationalitaet2","Prioritaet"]],data_mixed_effects2.Event, test_size=0.33, random_state=0)
#for i in cols_for_woe:
#    ce_target = ce.TargetEncoder(cols = [i])
#    x1_train[i]=ce_target.fit_transform(x1_train[i], y1_train).add_suffix('_woe')
#    x1_test[i]=ce_target.fit_transform(x1_test[i], y1_test).add_suffix('_woe')

    
    
#X_train = torch.from_numpy(x1_train.to_numpy()).float()
#y_train = torch.squeeze(torch.from_numpy(y1_train.to_numpy()).float())
#X_test = torch.from_numpy(x1_test.to_numpy()).float()
#y_test = torch.squeeze(torch.from_numpy(y1_test.to_numpy()).float())
#print(X_train.shape, y_train.shape)
#print(X_test.shape, y_test.shape)  



class binaryClassification(nn.Module):
  def __init__(self, input_shape):
    super(binaryClassification, self).__init__()
    self.fc1 = nn.Linear(input_shape, 25)
    self.fc2 = nn.Linear(25, 50)
    self.fc3 = nn.Linear(50, 10)
    self.fc4 = nn.Linear(10, 1)
    
    self.dropout = nn.Dropout(p=0.1)

  def forward(self, x):
    x = F.relu(self.fc1(x))
#    x = self.dropout(x)
    x = F.relu(self.fc2(x))
#    x = self.dropout(x)
    x = F.relu(self.fc3(x))
#    x = self.dropout(x)
    x= torch.sigmoid(self.fc4(x))
    return x

learning_rate = 0.01
epochs = 100

model = binaryClassification(input_shape=X_set.shape[1])
optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate)
#loss_fn = nn.BCELoss()


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


model.to(device)
print(model)
#criterion = nn.BCEWithLogitsLoss()
criterion = nn.BCELoss()

optimizer = optim.Adam(model.parameters(), lr=learning_rate)




model.train()
for e in range(1, epochs+1):
    epoch_loss = 0
    epoch_acc = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        
        y_pred = model(X_batch)
        
        loss = criterion(y_pred, y_batch.unsqueeze(1))
        acc = auroc(y_pred, y_batch.unsqueeze(1).type(torch.cuda.IntTensor))
        
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        

    print(f'Epoch {e+0:03}: | Loss: {epoch_loss/len(train_loader):.5f} | Acc: {epoch_acc/len(train_loader):.3f}')


y_pred_list = []
model.eval()
with torch.no_grad():
    for X_batch in test_loader:
        X_batch = X_batch.to(device)
        y_test_pred = model(X_batch)
        y_test_pred = torch.sigmoid(y_test_pred)
        y_pred_tag = torch.round(y_test_pred)
        y_pred_list.append(y_pred_tag.cpu().numpy())

y_pred_list = [a.squeeze().tolist() for a in y_pred_list]


y_pred_list = []
model.eval()
with torch.no_grad():
    for X_batch in test_loader:
        X_batch = X_batch.to(device)
        y_test_pred = model(X_batch)
        
        #y_test_pred = torch.sigmoid(y_test_pred)
        #y_pred_tag = torch.round(y_test_pred)
        y_pred_list.append(y_test_pred)

y_pred_list = [a.squeeze().tolist() for a in y_pred_list]
y_pred_list


#auroc(np.array(y_pred_list), np.array(y_set_test2))
from sklearn.metrics  import roc_auc_score
roc_auc_score(y_set_test, y_pred_list )

dataframe
ddd=dataframe[0:3000]
from scipy import stats

res_df=pd.DataFrame()
for i in ['Age2', 'N_reports','N_reports_total','Difference2', 'Len2']:
    
#variance is equal
    res_df.loc[1, i]=stats.levene(dataframe[dataframe.Event== 1].dropna()[i], dataframe[dataframe.Event== 0].dropna()[i]).pvalue
    #is normal distributed
    res_df.loc[2, i]=stats.kstest(dataframe[dataframe.Event== 1].dropna()[i],'norm').pvalue
    res_df.loc[3, i]=stats.kstest(dataframe[dataframe.Event== 0].dropna()[i],'norm').pvalue
    
    res_df.loc[4, i]=stats.ttest_ind(dataframe[dataframe.Event== 1].dropna()[i],dataframe[dataframe.Event== 0].dropna()[i]).pvalue
    res_df.loc[5, i]=stats.mannwhitneyu(dataframe[dataframe.Event== 1].dropna()[i],dataframe[dataframe.Event== 0].dropna()[i]).pvalue
    








