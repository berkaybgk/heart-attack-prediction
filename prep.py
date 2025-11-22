import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import OneHotEncoder
from imblearn.over_sampling import SMOTE
import pickle

def drop_unnecessary_columns(df: pd.DataFrame) -> pd.DataFrame:
    columns_to_remove = ['DeafOrHardOfHearing', 'BlindOrVisionDifficulty', 'DifficultyConcentrating', 'ChestScan', 'DifficultyDressingBathing', 'DifficultyErrands', 'HIVTesting', 'FluVaxLast12', 'PneumoVaxEver', 'TetanusLast10Tdap', 'HighRiskLastYear' , 'CovidPos']
    df = df.drop(columns=columns_to_remove, errors='ignore')
    return df

def preprocess_training_data(df: pd.DataFrame):

    # Remove outliers for BMI and SleepHours
    df = df[(df['BMI'] >= 10) & (df['BMI'] <= 60)]
    df = df[(df['SleepHours'] >= 2) & (df['SleepHours'] <= 13)]

    # Map 'Yes, but only during pregnancy (female)' to No
    # Map 'No, pre-diabetes or borderline diabetes' to Yes
    # For HadDiabetes column
    df['HadDiabetes'] = df['HadDiabetes'].replace({
        'Yes, but only during pregnancy (female)': 'No',
        'No, pre-diabetes or borderline diabetes': 'Yes'
    })

    # Yes/No columns
    yes_no_columns = ['PhysicalActivities', 'HadHeartAttack', 'HadAngina', 'HadStroke',
                      'HadAsthma', 'HadSkinCancer', 'HadCOPD', 'HadDepressiveDisorder',
                      'HadKidneyDisease', 'HadArthritis', 'DifficultyWalking', 'AlcoholDrinkers', 'HadDiabetes']
    for col in yes_no_columns:
        df[col] = df[col].map({'Yes': 1, 'No': 0})

    # Convert female/male to 1/0
    df['Sex'] = df['Sex'].map({'Female': 1, 'Male': 0})

    # Convert GeneralHealth to Poor=1, Fair=2, Good=3, Very good=4, Excellent=5
    health_mapping = {
        'Poor': 1,
        'Fair': 2,
        'Good': 3,
        'Very good': 4,
        'Excellent': 5
    }
    df['GeneralHealth'] = df['GeneralHealth'].map(health_mapping)

    # Map Heart Disease ratios for states, replace state names with ratios
    state_heart_attack = df.groupby('State')['HadHeartAttack'].mean().sort_values(ascending=False)
    state_ratio_mapping = {state: ratio for state, ratio in state_heart_attack.items()}
    df['State'] = df['State'].map(state_ratio_mapping)

    # One hot encode for multi-class categorical
    one_hot_columns = ['LastCheckupTime', 'RemovedTeeth', 'SmokerStatus', 'ECigaretteUsage', 'RaceEthnicityCategory', 'AgeCategory']
    # Use OneHotEncoder from sklearn
    one_hot_encoder = OneHotEncoder(sparse_output=False, drop='first')
    encoded_data = one_hot_encoder.fit_transform(df[one_hot_columns])
    encoded_df = pd.DataFrame(encoded_data, columns=one_hot_encoder.get_feature_names_out(one_hot_columns))
    df = df.reset_index(drop=True)
    encoded_df = encoded_df.reset_index(drop=True)
    df = pd.concat([df.drop(columns=one_hot_columns), encoded_df], axis=1)

    # Columns to use StandardScaler
    standard_columns = ['State', 'GeneralHealth', 'MentalHealthDays', 'PhysicalHealthDays', 'SleepHours', 'HeightInMeters', 'WeightInKilograms', 'BMI']
    standard_scaler = StandardScaler()
    df[standard_columns] = standard_scaler.fit_transform(df[standard_columns])
    df['HadHeartAttack'] = df['HadHeartAttack'].astype(int)

    # Handle class imbalance using SMOTE
    X = df.drop('HadHeartAttack', axis=1)
    y = df['HadHeartAttack']

    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    df = pd.concat([X_resampled, y_resampled], axis=1)

    # Return the cleaned DataFrame, Standard Scaler, One Hot Encoder, and state mapping
    return df, standard_scaler, one_hot_encoder, state_ratio_mapping


def preprocess_test_data(df: pd.DataFrame, standard_scaler, one_hot_encoder, state_ratio_mapping):
    """
    Preprocess test data using the fitted transformers from training data.
    Handles unknown categorical values to prevent encoding errors.
    """

    # Remove outliers for BMI and SleepHours (same as training)
    df = df[(df['BMI'] >= 10) & (df['BMI'] <= 60)]
    df = df[(df['SleepHours'] >= 2) & (df['SleepHours'] <= 13)]

    # Map HadDiabetes values
    df['HadDiabetes'] = df['HadDiabetes'].replace({
        'Yes, but only during pregnancy (female)': 'No',
        'No, pre-diabetes or borderline diabetes': 'Yes'
    })

    # Yes/No columns
    yes_no_columns = ['PhysicalActivities', 'HadHeartAttack', 'HadAngina', 'HadStroke',
                      'HadAsthma', 'HadSkinCancer', 'HadCOPD', 'HadDepressiveDisorder',
                      'HadKidneyDisease', 'HadArthritis', 'DifficultyWalking', 'AlcoholDrinkers', 'HadDiabetes']
    for col in yes_no_columns:
        df[col] = df[col].map({'Yes': 1, 'No': 0})

    # Convert female/male to 1/0
    df['Sex'] = df['Sex'].map({'Female': 1, 'Male': 0})

    # Convert GeneralHealth
    health_mapping = {
        'Poor': 1,
        'Fair': 2,
        'Good': 3,
        'Very good': 4,
        'Excellent': 5
    }
    df['GeneralHealth'] = df['GeneralHealth'].map(health_mapping)

    # Map states using the training state ratio mapping
    df['State'] = df['State'].map(state_ratio_mapping)
    # Handle any states not in training data (fill with mean of mapped values)
    state_mean = np.mean(list(state_ratio_mapping.values()))
    df['State'] = df['State'].fillna(state_mean)

    # One hot encode using fitted encoder - handle unknown categories
    one_hot_columns = ['LastCheckupTime', 'RemovedTeeth', 'SmokerStatus', 'ECigaretteUsage', 'RaceEthnicityCategory', 'AgeCategory']

    # Get the categories that the encoder was trained on
    trained_categories = one_hot_encoder.categories_

    # Replace any unknown categories with the most frequent category from training
    for i, col in enumerate(one_hot_columns):
        # Get valid categories for this column (first category since drop='first')
        valid_categories = trained_categories[i]
        most_frequent = valid_categories[0]  # First is typically most frequent

        # Replace any values not in valid categories with the most frequent one
        df[col] = df[col].apply(lambda x: x if x in valid_categories else most_frequent)

    # Now safely transform
    encoded_data = one_hot_encoder.transform(df[one_hot_columns])
    encoded_df = pd.DataFrame(encoded_data, columns=one_hot_encoder.get_feature_names_out(one_hot_columns))
    df = df.reset_index(drop=True)
    encoded_df = encoded_df.reset_index(drop=True)
    df = pd.concat([df.drop(columns=one_hot_columns), encoded_df], axis=1)

    # Apply StandardScaler using fitted scaler
    standard_columns = ['State', 'GeneralHealth', 'MentalHealthDays', 'PhysicalHealthDays', 'SleepHours', 'HeightInMeters', 'WeightInKilograms', 'BMI']
    df[standard_columns] = standard_scaler.transform(df[standard_columns])
    df['HadHeartAttack'] = df['HadHeartAttack'].astype(int)

    return df


if __name__ == "__main__":
    # Load and preprocess training data
    print("Loading training data...")
    df_train = pd.read_csv('./resources/heart_2022_train.csv')  # Changed from heart_2022_no_nans.csv
    df_train = drop_unnecessary_columns(df_train)

    print("Preprocessing training data...")
    df_train_processed, standard_scaler, one_hot_encoder, state_ratio_mapping = preprocess_training_data(df_train)

    # Save processed training data
    print("Saving processed training data...")
    df_train_processed.to_csv('./resources/heart_2022_train_processed.csv', index=False)

    # Save the fitted transformers
    print("Saving transformers...")
    with open('./resources/standard_scaler.pkl', 'wb') as f:
        pickle.dump(standard_scaler, f)

    with open('./resources/one_hot_encoder.pkl', 'wb') as f:
        pickle.dump(one_hot_encoder, f)

    with open('./resources/state_ratio_mapping.pkl', 'wb') as f:
        pickle.dump(state_ratio_mapping, f)

    print(f"Training data shape after preprocessing: {df_train_processed.shape}")
    print(f"Class distribution:\n{df_train_processed['HadHeartAttack'].value_counts()}")

    # Load and preprocess test data
    print("\nLoading test data...")
    df_test = pd.read_csv('./resources/heart_2022_test.csv')  # Changed filename
    df_test = drop_unnecessary_columns(df_test)

    print("Preprocessing test data...")
    df_test_processed = preprocess_test_data(df_test, standard_scaler, one_hot_encoder, state_ratio_mapping)

    print("Saving processed test data...")
    df_test_processed.to_csv('./resources/heart_2022_test_processed.csv', index=False)
    print(f"Test data shape after preprocessing: {df_test_processed.shape}")
    print(f"Test class distribution:\n{df_test_processed['HadHeartAttack'].value_counts()}")

    print("\nPreprocessing complete!")
