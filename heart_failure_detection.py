# -*- coding: utf-8 -*-
"""
Created on Sun Sep 19 13:04:28 2021

@author: delva
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import seaborn as sns

#Read in the raw data
data = pd.read_csv(r'C:\Users\delva\OneDrive\Desktop\Data Science\Projects\datasets\heart_failure_clinical_records_dataset.csv')

#Break data in half and separate into 2 csv files. i.e one X_full and another X_test.
train = data[0:149]
test = data[150:299]

#Remove comment hastag below to run these two commands if needed.
#train.to_csv('train.csv', index=False)
#test.to_csv('test.csv', index = False)

#---------------------------------------------------------------------------------------------------------------------------


#Scaterplot matrix of dataframe
#scatter = sns.pairplot(data, hue="DEATH_EVENT")


#---------------------------------------------------------------------------------------------------------------------------


from sklearn.model_selection import train_test_split

#Read in the 2 new datasets.
X_full = pd.read_csv(r"C:\Users\delva\OneDrive\Desktop\Data Science\Spyder\train.csv")
X_test_full = pd.read_csv(r"C:\Users\delva\OneDrive\Desktop\Data Science\Spyder\test.csv")

#Checks to see if datasets have any null values
#print(X_full.isnull().sum())
#print(X_test_full.isnull().sum())

#No missing values were detected so the line below is commented out
#X_full.dropna(axis=0, subset=['DEATH_EVENT'], inplace=True)


#Label the y prediction column and drop it from the X_full dataset 
y = X_full['DEATH_EVENT']
X_full.drop(['DEATH_EVENT'], axis = 1, inplace = True)


#Break off validation set from the training data
X_train_full, X_valid_full, y_train, y_valid = train_test_split(X_full, y,
                                                               train_size = 0.8, test_size = 0.2,
                                                               random_state = 0)

#Running X_full.dtypes() and X_test_full.dtypes will give us a look at the datatypes of each dataframe.
#We can see that all column/features have integer/float datatypes.
#The numeric_cols variable collects all the numeric columns. This isnt neccesary to run but its used as good practice to split into categorical/ numeric data types.
#The categorical_cols isnt needed either. Its just here for good practice.

numerical_cols = [cname for cname in  X_train_full.columns if
                  X_train_full[cname].dtype in ['int64','float64']]

#categorical_cols = [cname for cname in X_train_full.columns if
                    #X_train_full[cname].nunique() < 10 and 
                    #X_train_full[cname].dtype == "object"]

my_cols = numerical_cols #+ categorical_cols
X_train = X_train_full[my_cols].copy()
X_valid = X_valid_full[my_cols].copy()
X_test = X_test_full[my_cols].copy()

#---------------------------------------------------------------------------------------------------------------------------

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.model_selection import cross_val_score

#Manually picking which models to be used with general paramaters. This was based on current intuition.
model_1 = LogisticRegression(random_state=0)
model_2 = DecisionTreeClassifier(max_depth = 5, random_state = 0)
model_3 = RandomForestClassifier(n_estimators = 100, random_state=0)
model_4 = XGBClassifier(objective = 'binary:logistic', eval_metric = 'auc',
                            max_depth = 5, alpha = 10, learning_rate = 0.1,
                            n_estimators = 100)

score_names = ['accuracy','precision','recall','f1']

#Function used to test multiple classifiers and multiple metrics.
# Final result is a Dataframe of percentage values. Index is 4 models and columns are 4 metrics used.
def model_selection():
    model_list = [model_1, model_2, model_3,model_4]
    df = pd.DataFrame( columns = score_names)
    
    for model in model_list:
        metric_list = []
        for score in score_names:
            
            model.fit(X_train,y_train)
            preds = model.predict(X_valid)
            scores = cross_val_score(model, X_full, y, cv = 10,
                                scoring = score
                                ).mean()
            
            metric_list.append(scores)
        #print(metric_list)

        df.loc[len(df)] = metric_list
        df = df.rename( index={ 0 : 'Logistic Regression', 1 : 'Decision Tree', 2 : 'Random Forest', 3: 'XGBoost'})
    print(df)


model_selection()

#-------------------------------------------------------------------------------------------------------------------
#Based on the df Dataframe variable. Manually select the model with best metric scores.
#In this instance, the Logistic Model was the best.
best_model = model_1

#Generate test predictions
preds_test = best_model.predict(X_test)

#Output the predictions into a csv file.
output = pd.DataFrame({'Id': X_test.index,
                       'DEATH_EVENT': preds_test})
#output.to_csv('heart_failure_preds.csv', index=False)