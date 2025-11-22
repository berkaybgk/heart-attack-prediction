import pandas as pd
from sklearn.model_selection import train_test_split

# Load the original data
print("Loading data...")
df = pd.read_csv('./resources/heart_2022_no_nans.csv')

print(f"Original dataset shape: {df.shape}")
print(f"Class distribution:\n{df['HadHeartAttack'].value_counts()}")

# Split into train (80%) and test (20%) with stratification
print("\nSplitting data into train (80%) and test (20%)...")
df_train, df_test = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df['HadHeartAttack']  # Ensures balanced class distribution
)

# Save the splits
print("Saving train and test datasets...")
df_train.to_csv('./resources/heart_2022_train.csv', index=False)
df_test.to_csv('./resources/heart_2022_test.csv', index=False)

print(f"\nTrain set shape: {df_train.shape}")
print(f"Train class distribution:\n{df_train['HadHeartAttack'].value_counts()}")
print(f"\nTest set shape: {df_test.shape}")
print(f"Test class distribution:\n{df_test['HadHeartAttack'].value_counts()}")

print("\nTrain-test split complete!")
