# backend.py
# Import libraries
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import plotly.express as px
import re
import statsmodels.api as sm
import scipy.stats as stats
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV, validation_curve, cross_validate, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OrdinalEncoder
from numpy import log1p  # This is used for log transformation
from scipy.stats.contingency import association, chi2_contingency
from pandas.plotting import scatter_matrix
from sklearn.feature_selection import SelectFromModel
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, roc_auc_score, precision_recall_curve, roc_curve, make_scorer, auc
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from lightgbm import LGBMClassifier
from catboost import Pool, CatBoostClassifier
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler, SMOTE
from sklearn.datasets import make_classification
from sklearn.utils.class_weight import compute_class_weight

def load_data():
    data = pd.read_csv("https://raw.githubusercontent.com/data-bootcamp-v4/data/main/supermarket_sales.csv")
    data['Date'] = pd.to_datetime(data['Date'])
    return data

def get_summary(data):
    summary = pd.DataFrame({
        'Total Sales': [data['Total'].sum()],
        'Average Rating': [data['Rating'].mean()],
        'Total Transactions': [data['Invoice ID'].nunique()]
    })
    return summary

def plot_sales_over_time(data):
    data['Date'] = pd.to_datetime(data['Date'])
    sales_over_time = data.groupby(data['Date'].dt.date)['Total'].sum()
    plt.figure(figsize=(10, 5))
    plt.plot(sales_over_time.index, sales_over_time.values)
    plt.title('Sales Over Time')
    plt.xlabel('Date')
    plt.ylabel('Total Sales')
    plt.xticks(rotation=45)
    plt.tight_layout()
    return plt

