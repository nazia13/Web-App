import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split,cross_val_predict
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from scikitplot import metrics
# import scikitplot.metrics as mt
# from sklearn.metrics import plot_confusion_matrix
# import sklearn.metrics as metrics
import inspect, re
import xgboost
import io
import time
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
import timeit
import warnings
warnings.filterwarnings('ignore')

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity= 'all'




def data_cleaning(csv_file):
	tcc = pd.read_csv(csv_file)
	######### Data preprocessing to format feature columns and get prediction target column 'labels'

	category_cols = ['gender', 'SeniorCitizen', 'Partner', 'Dependents',
	                'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 
	                'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
	                'PaymentMethod']

	numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
	 

	target = 'Churn'
	ID_col = 'customerID'
	assert len(category_cols) + len(numeric_cols) + 2 == tcc.shape[1]


	tcc['TotalCharges']= tcc['TotalCharges'].apply(lambda x: x if x!= ' ' else np.nan).astype(float)
	tcc['MonthlyCharges'] = tcc['MonthlyCharges'].astype(float)

	tcc['TotalCharges'] = tcc['TotalCharges'].fillna(0)

	tcc['Churn'].replace(to_replace='Yes', value=1, inplace=True)
	tcc['Churn'].replace(to_replace='No',  value=0, inplace=True)

	features = tcc.drop(columns=[ID_col, target]).copy()
	labels = tcc['Churn'].copy()

	######### Split input data into training dataset, and holdout testing dataset

	X_train, X_test,y_train,y_test = train_test_split(features,labels, test_size=0.3, random_state=22)


	######### Apply ordinal encoder to convert categorical values into numerical values 

	ord_enc = OrdinalEncoder()
	ord_enc.fit(X_train[category_cols])

	X_train_OE = pd.DataFrame(ord_enc.transform(X_train[category_cols]), columns=category_cols)
	X_train_OE.index = X_train.index
	X_train_OE = pd.concat([X_train_OE, X_train[numeric_cols]], axis=1)

	X_test_OE = pd.DataFrame(ord_enc.transform(X_test[category_cols]), columns=category_cols)
	X_test_OE.index = X_test.index
	X_test_OE = pd.concat([X_test_OE, X_test[numeric_cols]], axis=1)


	return X_train, X_test, y_train, y_test, X_train_OE, X_test_OE


######### class of XGBoost model with functions including:  
######### model fitting, Hyperparameter tuning, target variable prediction, and output performance plots

class XGBmodel(object):
    
    def __init__(self,param):
        self.param=param
        
    def fit(self, X, y):
        clf = xgboost.XGBClassifier(random_state=22)
        clf_search=RandomizedSearchCV(estimator=clf,n_iter=50,param_distributions=self.param, scoring='accuracy',cv=10,n_jobs=4)
        clf_search.fit(X,y)
        self.clf = clf_search.best_estimator_
        return self
        
    def predict_proba(self, X):
        res_proba = self.clf.predict_proba(X)
        return res_proba
    
    def predict(self, X):
        res = self.clf.predict(X)
        return res
### display an download plots

    def show_test_result(self,X,y):
        result_proba=self.predict_proba(X)
        result_=self.predict(X)
        fig = metrics.plot_roc(y,result_proba)
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        global img_bytes
        img_bytes = buffer.read()
        st.pyplot()
          
        st.subheader('Confusion Matrix : ')
        predictions = cross_val_predict(self.clf, X, y)
        fig2 = metrics.plot_confusion_matrix(y, predictions, normalize=True)
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        global img_bytes2
        img_bytes2 = buffer.read()
        st.pyplot()

    def precision_recall_f1_visual(self,X,y):
    	print(classification_report(y,self.predict(X),digits=4))

    def download(self):

    	l=[]
    	global img_bytes
    	separator = "_"
    	for values in self.param.values():
    		l.append(values)

    	fname = "roc_plot_"  +  separator.join(map(str, l)) + ".png"
    	sname = "confusion_matrix" + separator.join(map(str, l))+ ".png"

    	with open(sname, 'wb') as f:
        	f.write(img_bytes2)

    	with open(fname, 'wb') as f:
        	f.write(img_bytes)

    	st.success('Plots downloaded successfully !')

