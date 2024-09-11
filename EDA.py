# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the datasets
train_df = pd.read_csv(r"C:\\Users\\hp\\OneDrive\\Desktop\\PRODIGY_DS_02\\train.csv")
test_df = pd.read_csv(r"C:\\Users\\hp\\OneDrive\\Desktop\\PRODIGY_DS_02\\test.csv")
gender_submission_df = pd.read_csv(r"C:\\Users\\hp\\OneDrive\\Desktop\\PRODIGY_DS_02\\gender_submission.csv")

# Display the first few rows of the train dataset
print(train_df.head())

# Data Cleaning
## Handle missing values
print("\nMissing values in each column:\n", train_df.isnull().sum())

# Fill missing values in 'Age' with the median age  
train_df['Age'].fillna(train_df['Age'].median(), inplace=True)

# Fill missing values in 'Embarked' with the most common value
train_df['Embarked'].fillna(train_df['Embarked'].mode()[0], inplace=True)

# Drop the 'Cabin' column due to too many missing values
train_df.drop(columns=['Cabin'], inplace=True)

## Convert 'Sex' and 'Embarked' into numerical values for analysis
train_df['Sex'] = train_df['Sex'].map({'male': 0, 'female': 1})
train_df['Embarked'] = train_df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

# Exploratory Data Analysis (EDA)
## Univariate Analysis
### Distribution of Survived
plt.figure(figsize=(8, 6))
sns.countplot(x='Survived', data=train_df)
plt.title('Distribution of Survival')
plt.show()

### Distribution of Age
plt.figure(figsize=(8, 6))
sns.histplot(train_df['Age'], kde=True, bins=30)
plt.title('Distribution of Age')
plt.show()

## Bivariate Analysis
### Survival Rate by Sex
plt.figure(figsize=(8, 6))
sns.barplot(x='Sex', y='Survived', data=train_df)
plt.title('Survival Rate by Sex')
plt.show()

### Survival Rate by Passenger Class
plt.figure(figsize=(8, 6))
sns.barplot(x='Pclass', y='Survived', data=train_df)
plt.title('Survival Rate by Passenger Class')
plt.show()

### Age vs. Fare by Survival Status
plt.figure(figsize=(10, 8))
sns.scatterplot(x='Age', y='Fare', hue='Survived', data=train_df)
plt.title('Age vs. Fare by Survival Status')
plt.show()

## Correlation Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(train_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()

# Summarize Findings
print("\nKey Findings:")
print("1. Females had a higher survival rate compared to males.")
print("2. Passengers in higher classes (1st class) had better survival chances.")
print("3. Younger passengers had a slightly higher survival rate, and those who paid higher fares also had better survival chances.")

# Prepare the data for modeling 
## Drop unnecessary columns
train_df.drop(columns=['Name', 'Ticket', 'PassengerId'], inplace=True)

# Save the cleaned dataset for further use
train_df.to_csv(r"C:\\Users\\hp\\OneDrive\\Desktop\\PRODIGY_DS_02\\cleaned_train.csv", index=False)
