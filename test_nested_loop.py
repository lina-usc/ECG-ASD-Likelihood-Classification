#!/usr/bin/env python
# coding: utf-8

# In[14]:


#!/usr/bin/env python
# coding: utf-8

# In[1]:
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score     
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_validate



import pickle
from datetime import datetime


#factrozie labels import numpy as np
import os
import math
import pandas as pd
from path import Path
from sklearn.model_selection import ShuffleSplit

from sklearn.ensemble import GradientBoostingClassifier, ExtraTreesClassifier, RandomForestClassifier,AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
import xgboost as xgb



# In[3]:


def fun_read_csv(file_name):

    segment_len = file_name.split('_')[1]
    #print(segment_len)
    df_for_classification = pd.read_csv(file_name)
    y = df_for_classification[['Labels']]
    X=df_for_classification.drop(columns = ['MeanNN','MaxNN', 'MinNN','Labels'])
    #LL 0 , EL 1
    return X , y, segment_len
    

model_params = { 
    #1
        'RF':{ 'model':RandomForestClassifier(random_state=2),
                'parms': {'n_estimators': [1,10,100,200],
          'max_depth': [1,5,10,20], 
          'max_features': ['sqrt','log2'],
          'min_samples_split': [2,5,10], 'n_jobs': [-1]}
             },
        
        #2
    'GB':{ 'model':GradientBoostingClassifier(random_state=3000),
            'parms':  {'n_estimators': [1,10,20,30], 
                       'learning_rate' : [0.01,0.05,0.1,0.5],
           'subsample' : [0.1,0.5,1.0], 'max_depth': [1,3,5,10,20,30]}},
     #3   
    'DT': { 'model':DecisionTreeClassifier(random_state=2),
        'parms':   {'criterion': ['gini', 'entropy'], 
           'max_depth': [1,5,10,20],
           'max_features': [2,5,7],
           'min_samples_split': [2,5,10]}
          },
        
     #4   
    'ET': { 'model': ExtraTreesClassifier(random_state=0),
            'parms':   { 'n_estimators': [1,10,100], 
           'criterion' : ['gini', 'entropy'] ,
           'max_depth': [1,5,10,20,], 
           'max_features': ['sqrt','log2'],
           'min_samples_split': [2,5,10],
           'n_jobs': [-1]}
          },
     #5   
    'KNN' :{ 'model': KNeighborsClassifier(),
            'parms':  {'n_neighbors': [1,5,10],
            'weights': ['uniform','distance'],
            'algorithm': ['auto','ball_tree','kd_tree']}
           },
    
    #6
    'ADB' : { 'model' :AdaBoostClassifier(random_state=0),
            'parms' :  {'algorithm': ['SAMME', 'SAMME.R'],
                       'n_estimators': [1,10,100]}
              
              },
    #7
    'XGB' : { 'model' :xgb.XGBClassifier(),

            'parms': {
                'max_depth': range (2, 10, 1),
                'n_estimators': range(60, 220, 40),
                'learning_rate': [0.1, 0.01, 0.05]
                            }
            },
    #8         
    'MLP' :  { 'model' : MLPClassifier(max_iter= 1000000, random_state=0),
                 
            'parms' :  {'hidden_layer_sizes': [(100,200,), (10)],
                'activation': ['tanh', 'relu', 'logistic'],
                'solver': ['sgd', 'adam', 'lbfgs'],
                'alpha': [0, 0.0001, 0.05],
                'learning_rate': ['constant','adaptive']}  
                 
             }
}



def nested_loop(X, y, innerSplits,  outerSplits, num_trials, mp):
    
    scoring = {
    'accuracy': make_scorer(accuracy_score),
    'precision': make_scorer(precision_score, average='weighted'),
    'recall': make_scorer(recall_score, average='weighted'),
    'f1_score': make_scorer(f1_score, average='weighted'),
    'roc' : make_scorer(roc_auc_score, average='weighted')
    }
    # Arrays to store scores
    #non_nested_scores = np.zeros(NUM_TRIALS)
    #nested_scores = {}

    # Loop for each trial
    for i in range(num_trials):

        inner_cv = KFold(n_splits=innerSplits, shuffle=True, random_state=i)
        outer_cv = ShuffleSplit(n_splits = outerSplits, train_size = 0.8, test_size = 0.2, random_state = 0)
        
        """
        #Giving error on scoring function.
        # Non_nested parameter search and scoring
        clf = GridSearchCV(mp['model'], mp['parms'], cv=outer_cv, scoring=scoring)
        clf.fit(X, y.squeeze())
        non_nested_scores[i] = clf.best_score_
        
        """
        
        
        # Nested CV with parameter optimization
        clf = GridSearchCV(mp['model'], mp['parms'], cv=inner_cv)
        #print(clf)
        
        nested = cross_validate(clf, X=X, y=y.squeeze(), cv=outer_cv, scoring= scoring )
        
        #return nested_scores, non_nested_scores
        return nested


# In[7]:


def get_nested_results(i, innerSplits,  outerSplits, num_trials, mp):

    X, y, seg_len = fun_read_csv(i)
    for model_name, mp in model_params.items():
        
        #nested, non_nested = nested_loop(X, y, innerSplits,  outerSplits, num_trials, mp)
        nested = nested_loop(X, y, innerSplits,  outerSplits, num_trials, mp)
        
        return seg_len ,nested,
    
             #'Non_Nested_Scores':non_nested}    



  
start_time = datetime.now()

# REading files which are extracted from the preprocessing part

all_segments_csvs =[f for f in os.listdir() if f.endswith('classification2023.csv')]
print(all_segments_csvs)



num_trials = 1
outerSplits = 100
innerSplits = 5


# do your work here
results = []
for segments in tqdm(all_segments_csvs): 
    print(segments)
    for model_name, mp in model_params.items():
       
        print(model_name)
        seg_len, scores = get_nested_results(segments, innerSplits,  outerSplits, num_trials, mp)
        results.append([seg_len,scores, model_name])
                      
# create a binary pickle file 
f = open("test_nested.pkl","wb")

# write the python object (dict) to pickle file
pickle.dump(results,f)

# close file
f.close()  

end_time = datetime.now()
print(end_time - start_time)


# In[ ]:




