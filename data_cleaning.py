import pandas as pd
def pre_process(df):    
    #level 1
    cgpa_mean = df["CGPA"].mean(skipna=True) # Calculate the mean CGPA (excluding missing values)
    df_clean = df.dropna(subset=["Substance_Use"]).copy() # Remove rows with missing values in Substance_Use
    df_clean["CGPA"] = df_clean["CGPA"].fillna(cgpa_mean) # Fill missing CGPA values with the calculated mean

    #level 2- Feature Engineering: Create a binary variable 'Is_STEM'
    # Assign 1 for Engineering, Medical, and Computer Science; 0 for all others
    stem_courses = ['Engineering', 'Medical', 'Computer Science']
    df_clean['Is_STEM'] = df_clean['Course'].isin(stem_courses).astype(int)

    # level 3- Data Transformation: Convert categorical variables to numerical values
    # Define mapping: Poor/Low = 1, Average/Moderate = 2, Good/High = 3
    sleep_mapping = {'Poor': 1, 'Average': 2, 'Good': 3}
    support_mapping = {'Low': 1, 'Moderate': 2, 'High': 3}
    physical_mapping= {'Low': 1, 'Moderate': 2, 'High': 3}
    diet_mapping = {'Poor': 1, 'Average': 2, 'Good': 3}
    counseling_mapping= {'Never':1, 'Occasionally':2,'Frequently':3}
    substance_mapping= {'Never':1, 'Occasionally':2,'Frequently':3}

    # Apply the mapping to the respective columns
    df_clean['Sleep_Quality'] = df_clean['Sleep_Quality'].replace(sleep_mapping)
    df_clean['Social_Support'] = df_clean['Social_Support'].replace(support_mapping)
    df_clean['Physical_Activity']=df_clean['Physical_Activity'].replace(physical_mapping)
    df_clean['Diet_Quality']=df_clean['Diet_Quality'].replace(diet_mapping)
    df_clean['Counseling_Service_Use']=df_clean['Counseling_Service_Use'].replace(counseling_mapping)
    df_clean['Substance_Use']=df_clean['Substance_Use'].replace(substance_mapping)


    # level 4a- Handling Outliers using IQR for continuous variables
    # This will remove extreme/unrealistic values for Age, CGPA, and Credit Load
    outlier_columns = ['Age', 'CGPA', 'Semester_Credit_Load']

    for col in outlier_columns:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        # Filter the data to keep only values within the calculated bounds
        df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]

    # level 4b- Logical Range Validation for score columns
    # Ensure that scores like Stress, Depression, and Anxiety are within the valid range [0, 5]
    score_columns = ['Stress_Level', 'Depression_Score', 'Anxiety_Score', 'Financial_Stress']

    for col in score_columns:
        # Filter out any values that are negative or greater than 5
        df_clean = df_clean[(df_clean[col] >= 0) & (df_clean[col] <= 5)]
   # Save the final cleaned dataset as st1.csv
    df_clean.to_csv("clean_data.csv", index=False)
    return df_clean

        

