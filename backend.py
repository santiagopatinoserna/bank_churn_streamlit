# backend.py
# Import libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math
import plotly.express as px
import re
import statsmodels.api as sm
import scipy.stats as stats
import streamlit as st
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


# Function to load and analyze data
def load_and_analyze_data(file_path):
    # 1.2. Load the dataset
    df = pd.read_csv(file_path)

    # 1.3. Check the shape of the dataset, duplicated rows, and statistics summary
    def initial_data_checking(df):
        # Print the shape of the DataFrame (number of rows and columns)
        print(f"1. Shape of the DataFrame: {df.shape}")

        # Print the count of duplicate rows
        print(f"2. Duplicate Rows Number: {df.duplicated().sum()}")

        # Print summary statistics for numerical columns
        print("3. Summary Statistics:")
        return pd.DataFrame(df.describe())

    initial_check_summary = initial_data_checking(df)
    print(initial_check_summary)  # Print summary statistics

    # 1.4. Assess data quality: datatypes, number and % of unique values and missing values
    def unique_and_missing_values_dtype(df):
        # Non-null counts and data types
        non_null_counts = df.notnull().sum()
        dtypes = df.dtypes

        # Count of unique values
        unique_count = df.nunique()

        # Percentage of unique values
        unique_percentage = (df.nunique() / len(df)) * 100

        # Count of missing values
        missing_count = df.isnull().sum()

        # Percentage of missing values
        missing_percentage = df.isnull().mean() * 100

        # Combine into a DataFrame
        summary = pd.DataFrame({
            'non-Null_count': non_null_counts,
            'dtype': dtypes,
            'unique_values': unique_count,
            '%_unique': unique_percentage.round(2).astype(str) + '%',
            'missing_values': missing_count,
            '%_missing': missing_percentage.round(2).astype(str) + '%'
        })

        return summary

    data_quality_summary = unique_and_missing_values_dtype(df)
    print("\n4. Data Quality Summary:")
    print(data_quality_summary)  # Print data quality summary

    # 1.5. Identify categorical variables from numerical formats (less than 20 unique values)
    potential_categorical_from_numerical = df.select_dtypes(
        "number").loc[:, df.select_dtypes("number").nunique() < 20]
    print("\n5. Potential Categorical Variables from Numerical Columns:")
    # Print potential categorical variables
    print(potential_categorical_from_numerical.head())

    return df, initial_check_summary, data_quality_summary, potential_categorical_from_numerical


# Function to clean and format the DataFrame
def clean_and_format_dataframe(df, integer_columns):
    # 2.1. Delete Duplicates
    initial_row_count = df.shape[0]
    df.dropna(inplace=True)
    final_row_count = df.shape[0]
    print(f"2.1.Deleted {initial_row_count - final_row_count} "
          f"duplicate/missing rows.")

    # 2.2. Standardize Column Names
    def clean_column(name):
        # Convert camel case to snake case
        name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
        name = re.sub('([a-z0-9])([A-Z])', r'\1_\2', name)
        name = name.strip()  # Remove leading and trailing spaces
        # Replace non-alphanumeric characters with underscores
        name = re.sub(r'[^0-9a-zA-Z]+', '_', name)
        # Replace multiple underscores with a single underscore
        name = re.sub(r'_+', '_', name)
        name = name.lower()  # Convert to lowercase
        return name.strip('_')  # Remove leading and trailing underscores

    df.columns = [clean_column(col) for col in df.columns]
    print(f"2.2.Standardized column names: {df.columns.tolist()}")

    # 2.3. Data Types Correction
    def convert_float_to_integer(df, columns):
        for column in columns:
            df[column] = df[column].astype(int)
        return df

    df_cleaned = convert_float_to_integer(df, integer_columns)
    print(f"2.3.Converted columns to integer: {integer_columns}")

    # Export cleaned dataset
    df_cleaned_path = './df_cleaned.csv'
    df.to_csv(df_cleaned_path, index=False)
    print(df_cleaned.head())

    return df_cleaned, df_cleaned_path


# Function for Univariate Analysis
def univariate_analysis(df_cleaned):
    # 3.1. Separate categorical and numerical columns
    categorical_columns = ['geography', 'gender', 'tenure',
                           'num_of_products', 'has_cr_card', 'is_active_member', 'exited']
    numerical_columns = ['credit_score', 'age', 'balance', 'estimated_salary']

    df_categorical = df_cleaned[categorical_columns]
    df_numerical = df_cleaned[numerical_columns]

    # 3.2. Categorical variables. Frequency tables: counts and proportions
    def generate_frequency_proportion_tables(df_categorical):
        frequency_proportion_results = {}

        for col in df_categorical.columns:
            # Calculate frequency and proportion
            frequency = df_categorical[col].value_counts()
            proportion = df_categorical[col].value_counts(
                normalize=True).round(2)

            # Combine into a single DataFrame
            result_table = pd.DataFrame({
                'Frequency': frequency,
                'Proportion': proportion
            })

            # Add the result to the dictionary
            frequency_proportion_results[col] = result_table

        return frequency_proportion_results

    # 3.3. Categorical variables. Barplots
    def plot_categorical_barplots(df_categorical):
        num_cols = 3
        num_plots = len(df_categorical.columns)
        num_rows = math.ceil(num_plots / num_cols)

        fig, ax = plt.subplots(
            num_rows, num_cols, figsize=(num_cols * 5, num_rows * 5))
        ax = ax.flatten()

        for i, col in enumerate(df_categorical.columns):
            sns.countplot(data=df_categorical, x=col, ax=ax[i])
            ax[i].set_title(f'Distribution of {col}')
            ax[i].set_xlabel(col)
            ax[i].set_ylabel('Count')
            plt.xticks(rotation=45)

        plt.tight_layout()
        return fig

    # 3.4. Categorical variables. Pie charts
    def plot_categorical_pie_charts(df_categorical):
        num_cols = 3
        num_plots = len(df_categorical.columns)
        num_rows = math.ceil(num_plots / num_cols)

        fig, ax = plt.subplots(
            num_rows, num_cols, figsize=(num_cols * 5, num_rows * 5))
        ax = ax.flatten()

        for i, col in enumerate(df_categorical.columns):
            df_categorical[col].value_counts().plot.pie(
                autopct='%1.1f%%', colors=sns.color_palette("Set3"), startangle=90, ax=ax[i])
            ax[i].set_title(f'Distribution of {col}')
            ax[i].set_ylabel('')  # Hide the y-label for better aesthetics

        plt.tight_layout()
        return fig

    # 3.5. Numerical variables. Summary Statistics
    def summary_statistics(df_numerical):
        return pd.DataFrame(df_numerical.describe())

    # 3.6. Numerical variables. Shape of the distribution: Skewness and Kurtosis
    def calculate_skewness_kurtosis(df_numerical):
        results = {'Column': [], 'Skewness': [], 'Kurtosis': []}

        for column in df_numerical.columns:
            skewness = round(df_numerical[column].skew(), 2)
            kurtosis = round(df_numerical[column].kurtosis(), 2)

            results['Column'].append(column)
            results['Skewness'].append(skewness)
            results['Kurtosis'].append(kurtosis)

        return pd.DataFrame(results)

    # 3.7. Plot Histograms for Numerical Variables
    def plot_histograms(df_numerical):
        num_cols = 2
        num_plots = len(df_numerical.columns)
        num_rows = math.ceil(num_plots / num_cols)

        fig, ax = plt.subplots(num_rows, num_cols, figsize=(15, num_rows * 5))
        ax = ax.flatten()

        for i, col in enumerate(df_numerical.columns):
            df_numerical[col].plot.hist(
                bins=60, ax=ax[i], color="skyblue", edgecolor="black")
            ax[i].set_title(f'Distribution of {col}')
            ax[i].set_xlabel(col)
            ax[i].set_ylabel('Frequency')

        plt.tight_layout()
        return fig

    # 3.8. Plot Boxplots for Numerical Variables
    def plot_boxplots(df_numerical):
        num_cols = 2
        num_rows = math.ceil(len(df_numerical.columns) / num_cols)

        fig, ax = plt.subplots(num_rows, num_cols, figsize=(15, num_rows * 5))
        ax = ax.flatten()

        for i, column in enumerate(df_numerical.columns):
            sns.boxplot(data=df_numerical[column], ax=ax[i], color="lightblue")
            ax[i].set_title(f'Boxplot of {column}')

        plt.tight_layout()
        return fig

    # Call all functions and display results in Streamlit
    st.subheader("3.1. Define Categorical and Numerical Variables")
    st.write(f"Categorical Variables: {df_categorical.columns}.")
    st.write(f"Numerical Variables: {df_numerical.columns}.")

    st.subheader(
        "3.2. Frequency and Proportion Tables for Categorical Variables")
    frequency_proportion_tables = generate_frequency_proportion_tables(
        df_categorical)
    for col, table in frequency_proportion_tables.items():
        st.write(f"{col}:\n")
        st.table(table)

    st.subheader("3.3. Plot Categorical Barplots")
    st.pyplot(plot_categorical_barplots(df_categorical))

    st.subheader("3.4. Plot Categorical Pie Charts")
    st.pyplot(plot_categorical_pie_charts(df_categorical))

    st.subheader("3.5. Summary Statistics for Numerical Variables")
    summary_stats = summary_statistics(df_numerical)
    st.table(summary_stats)

    st.subheader("3.6. Skewness and Kurtosis for Numerical Variables")
    skewness_kurtosis = calculate_skewness_kurtosis(df_numerical)
    st.table(skewness_kurtosis)

    st.subheader("3.7. Plot Histograms for Numerical Variables")
    st.pyplot(plot_histograms(df_numerical))

    st.subheader("3.8. Plot Boxplots")
    st.pyplot(plot_boxplots(df_numerical))
