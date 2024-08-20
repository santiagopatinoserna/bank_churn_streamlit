# app.py
import streamlit as st
from backend import load_and_analyze_data, clean_and_format_dataframe, univariate_analysis
import sys
print("Python version:", sys.version)

try:
    import seaborn as sns
    print("Seaborn is available.")
except ModuleNotFoundError as e:
    print("Seaborn not found:", e)


print("Import successful!")


def main():
    st.title('Bank Churn Prediction Analysis')

    # Sidebar navigation
    st.sidebar.title('Navigation')
    page = st.sidebar.radio("Select a Step", [
                            "Data Load & Analysis", "Clean & Format DataFrame", "Univariate Analysis"])

    # Load data
    data_url = 'https://raw.githubusercontent.com/NGravereaux/bank_churn_streamlit/main/bank_churn_prediction_modeling.csv'
    df, initial_check_summary, data_quality_summary, potential_categorical_from_numerical = load_and_analyze_data(
        data_url)

    # Define df_cleaned in the outer scope
    df_cleaned = None

    if page == "Data Load & Analysis":
        st.header("Data Load & Initial Analysis")

        # Display initial data analysis results
        st.subheader("Shape of the DataFrame and Duplicate Rows")
        st.write(f"Shape of the DataFrame: {df.shape}")
        st.write(f"Duplicate Rows Number: {df.duplicated().sum()}")

        st.subheader("Summary Statistics")
        st.table(initial_check_summary)

        st.subheader("Data Quality Summary")
        st.table(data_quality_summary)

        st.subheader("Potential Categorical Variables from Numerical Columns")
        st.dataframe(potential_categorical_from_numerical)

    elif page == "Clean & Format DataFrame":
        st.header("Clean & Format DataFrame")

        # Clean and format the dataframe
        integer_columns = ['age', 'balance', 'has_cr_card',
                           'is_active_member', 'estimated_salary']
        df_cleaned, df_cleaned_path = clean_and_format_dataframe(
            df, integer_columns)

        st.subheader("Cleaned DataFrame")
        st.dataframe(df_cleaned.head())

        st.subheader("Path to Cleaned CSV")
        st.write(df_cleaned_path)

    elif page == "Univariate Analysis":
        st.header("Univariate Analysis")

        # Check if df_cleaned is available, if not, clean the data
        if df_cleaned is None:
            integer_columns = ['age', 'balance', 'has_cr_card',
                               'is_active_member', 'estimated_salary']
            df_cleaned, _ = clean_and_format_dataframe(df, integer_columns)

        # Perform univariate analysis
        univariate_analysis(df_cleaned)


if __name__ == '__main__':
    main()
