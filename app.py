import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, plot_confusion_matrix, roc_curve, plot_roc_curve
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from xgb import data_cleaning
from xgb import XGBmodel
import xgboost
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from scikitplot import metrics 
import inspect, re
import time
import timeit
st.write("""
# Churn Rate Prediction
""")

st.sidebar.header('User Input Parameters')

uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")
if uploaded_file is None:
	st.write('Please upload a file to start') 
else:
	X_train, X_test, y_train, y_test, X_train_OE, X_test_OE = data_cleaning(uploaded_file)


def user_input_features():
		  
	learning_rate = st.sidebar.selectbox('Select Learning Rate', [0.01, 0.016, 0.027, 0.046, 0.077, 0.129, 0.215, 0.359])
	max_depth = st.sidebar.selectbox('Select Max Depth', [5, 10, 15])
	n_estimators = st.sidebar.selectbox('Select Number of Estimators', [50, 70, 90, 110, 130, 150])
	colsample_bytree = st.sidebar.selectbox('Select Colsample Bytree', [0.4, 0.6, 0.8])

	param = {'learning_rate': learning_rate,
	    'max_depth': max_depth,
	    'n_estimators': n_estimators,
	    'colsample_bytree': colsample_bytree}

	return param


param = user_input_features()
for k, v in param.items():
	param[k] = [v]
print(param)
# st.subheader('User Input parameters')
# st.write(df)

xgb_param = dict(learning_rate=[0.01, 0.016, 0.027, 0.046, 0.077, 0.129, 0.215, 0.359],
                     max_depth=[5, 10, 15],
                     n_estimators=[ 50,  70,  90, 110, 130, 150],
                     colsample_bytree=[0.4,  0.6,  0.8]
                     )

if st.button('Fit Model'):
	start = timeit.default_timer()
	xgb_model= XGBmodel(param)
	xgb_model.fit(X_train_OE,y_train)
	stop = timeit.default_timer()
	st.write('using Time:{:.2f} s'.format(stop - start))
	xgb_model.precision_recall_f1_visual(X_test_OE,y_test)

	st.subheader('Results')
	xgb_model.show_test_result(X_test_OE,y_test)

if st.sidebar.button('Download Plots'):
	xgb_model= XGBmodel(param)
	xgb_model.download()






	
	

