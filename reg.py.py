# -*- coding: utf-8 -*-
"""
Created on Sun Sep 11 18:01:37 2016

@author: Deigant
"""


import pandas as pd
import numpy as np
from sklearn import feature_selection
from pandas import DataFrame
import scipy
from sklearn import preprocessing
from keras.models import Sequential
import keras

import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.cross_validation import StratifiedKFold
from sklearn.feature_selection import RFECV


from sklearn.ensemble import ExtraTreesClassifier
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectFromModel

from sklearn.datasets import make_gaussian_quantiles
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

import math
import time



#import data from excel
a = [];
data = pd.read_excel('Book1.xlsx', '2732', index_col=None, na_values=['NA'])
data = data.fillna(0);

a = data.as_matrix()
serial = np.arange(len(a[0]));


a = np.vstack((a,serial));

a = scipy.delete(a,0,1); #remove sl no
a = scipy.delete(a,0,1); #remove hospital number
a = scipy.delete(a,0,1); #remove DOA

#clean the data from wrong or unrequired data
k = len(a);
m = len(a[0]);
i = 0;
j = 0;
while i<k:
    j=0;
    while j<m:
        if(a[i][j] == u'>5'or a[i][j] == u'>80' or a[i][j] == ">80" or a[i][j] == u'>30'):
            a = scipy.delete(a,i,0)
            k = k-1;
            i = i-1
        elif a[i][j] == u'29..4':
            a[i][j] = 29.4
        elif a[i][j]== u'-' or a[i][j] == '5.2.8':
            a = scipy.delete(a,i,0);
            i = i -1;
            k = k-1;
        elif a[i][j] == u'2_3':
            a[i][j] = 2.5;
        elif type(a[i][j])==unicode:
            # print a[i][j]
            a = scipy.delete(a,j,1);
            j = j-1;
            m = m-1;
        j = j+1;
        
    i = i+1;
array_2 = np.array([]);
final_indexed_array = np.vstack(a[:,:]).astype(np.float);
final_array = scipy.delete(final_indexed_array,len(final_indexed_array)-1,0);
f = final_array;
class_array = f;

#divide the data into 2 categories and form the labels for the data [for 2 categoris los<=7 and los>7]
i =0;
m = len(final_indexed_array);
count =0;

label = [] ;

while i<m:
    if(final_indexed_array[i][0]>7):
        label.append(1)
        if(count == 0 ):
            count +=1;
            array_2 = final_indexed_array[i,:];
        else:
            array_2 = np.vstack([array_2,final_indexed_array[i,:]]);        
        final_indexed_array = scipy.delete(final_indexed_array,i,0);
        i = i-1; 
        m = m-1;
    else:
        label.append(0);
    i = i+1;
    
    
label.pop();


main_target_2 = array_2[:,0];
array_2 = scipy.delete(array_2,0,1);
final_array = scipy.delete(final_indexed_array,len(final_indexed_array)-1,0);
main_target = final_array[:,0];
final_array = scipy.delete(final_array,0,1);
f = scipy.delete(f,0,1); # matrix that contains all the data
main_target = np.array(main_target) #los

# extract the import feature using ensemble trees
clf = ExtraTreesClassifier()
clf = clf.fit(final_array, main_target)
clf.feature_importances_  
model = SelectFromModel(clf, prefit=True)
final_array = model.transform(final_array)
f= model.transform(f);
array_2 = model.transform(array_2);


g = main_target[int(0.8*len(final_array)):len(main_target)];

var = np.var(main_target);
mean = np.mean(main_target);

classify = 01; # variable to contain the predicted class of the data for the los is required


label = np.array(label)

#raw data for classifier
predict_train_data = f[0:int(0.8*len(f)), 0:len(f[0])]
predict_train_target = label.reshape(-1,1)[0:int(0.8*len(label)),0];


predict_test_data = f[int(0.8*len(f)):len(f), 0:len(f[0])]
predict_test_target =label.reshape(-1,1)[int(0.8*len(label)):len(label),0];


# ADABoost classifier for classfication

bdt_real = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=2),
    n_estimators=600,
    learning_rate=1)

bdt_discrete = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=2),
    n_estimators=600,
    learning_rate=1.5,
    algorithm="SAMME")
    
bdt_real.fit(predict_train_data,predict_train_target )
bdt_discrete.fit(predict_train_data, predict_train_target)
real_test_errors = []
discrete_test_errors = []
weight =clf.feature_importances_*100;

for i in range(len(weight)):
    if weight[i]<4:
        weight[i]= 1;


clf_2 = ExtraTreesClassifier(n_estimators=10, max_depth=None,min_samples_split=1, random_state=0,class_weight =weight)
clf_2 = clf.fit(predict_train_data,predict_train_target)

from sklearn.neighbors import KNeighborsClassifier

neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(predict_train_data, predict_train_target)
 
from sklearn.lda import LDA
clf_3 = LDA()
clf_3.fit(predict_train_data, predict_train_target)
LDA(n_components=None, priors=None, shrinkage=None, solver='svd',
  store_covariance=False, tol=0.0001)
  
from sklearn import tree

clf_4 = tree.DecisionTreeClassifier()
clf_4 = clf.fit(predict_train_data, predict_train_target)

####

#training the first neural network
#for Category 1 [LOS<=7]
train_data = final_array[0:int(0.8*len(final_array)), 0:len(final_array[0])]
train_target = main_target.reshape(-1,1)[0:int(0.8*len(final_array)),0];


test_data = final_array[int(0.8*len(final_array)):len(final_array), 0:len(final_array[0])]
test_target =main_target.reshape(-1,1)[int(0.8*len(final_array)):len(final_array),0];
 

model = Sequential()
model.add(keras.layers.core.Dense(len(train_data[0]), input_dim=len(train_data[0]),init='uniform',activation = 'relu',bias = True))
model.add(keras.layers.core.Dense(8, init='uniform', activation='relu',bias = True))
model.add(keras.layers.core.Dense(1,init = 'uniform',bias = True))
model.compile(loss='mean_squared_error', optimizer='adam')
keras.layers.core.Dropout(0.1)
model.fit(train_data, train_target, nb_epoch=150, batch_size=10)
model.evaluate(train_data, train_target,batch_size = 10)

#training the 2nd Neural network
#For category II LOS>7

#array_2 = scipy.delete(array_2,0,1);
train_data_2 = array_2[0:int(0.9*len(array_2)), 0:len(array_2[0])]
train_target_2 = main_target_2.reshape(-1,1)[0:int(0.9*len(array_2)),0];

test_data_2 = array_2[int(0.9*len(array_2)):len(array_2),:]
test_target_2 =main_target_2.reshape(-1,1)[int(0.9*len(array_2)):len(array_2),0];
        
model_2 = Sequential()
model_2.add(keras.layers.core.Dense(len(train_data_2[0]),input_dim=len(train_data_2[0]),init='uniform',activation = 'relu',bias = True))
model_2.add(keras.layers.core.Dense(8, init='uniform', activation='relu',bias = True))
model_2.add(keras.layers.core.Dense(1,init = 'uniform',bias = True))

model_2.compile(loss='mean_squared_error', optimizer='adam')
keras.layers.core.Dropout(0.1)
model_2.fit(train_data_2, train_target_2, nb_epoch=500, batch_size=5)
model_2.evaluate(train_data_2, train_target_2,batch_size = 5)


final_model_test_data = np.vstack((test_data,test_data_2));
final_model_test_target = np.concatenate((test_target,test_target_2),axis = 0);
 

r = [];
m = [];


#ans = bdt_discrete.predict(final_model_test_data);
ans = clf_2.predict(final_model_test_data)

#ans = clf_3.predict(final_model_test_data);
#ans = clf_4.predict(final_model_test_data);

for mn in range(len(ans)):
    if(ans[mn]==0):
        classify =0;
    else:
        classify = 1;
##############################1ST######################################################
    if (classify == 0):
        q = np.matrix([final_model_test_data[mn]])
        predictions = model.predict(q);
        r.append(predictions);
#######################################################################################
    elif classify == 1:
        q = np.matrix([final_model_test_data[mn]])
        predictions = model_2.predict(q);
        r.append(predictions);        
#######################################################################################

predicted_results = r;

for t in range(len(r)):
    if (r[t] - int(r[t]))>0.5:
        predicted_results[t] = math.ceil(r[t]*10/10);
    else:
        predicted_results[t] = int(r[t]);
        predicted_results = r;
"""
predicted_results_2 = m;
for t in range(len(m)):
    if (m[t] - int(m[t]))>0.5:
        predicted_results_2[t] = math.ceil(m[t]*10/10);
    else:
        predicted_results_2[t] = int(m[t]);
    """
mean_squared_error = np.mean(np.absolute(np.array(predicted_results) - final_model_test_target)**2)
print 
print mean_squared_error;

