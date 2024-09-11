Titanic Dataset Analysis: Data Cleaning and Exploratory Data Analysis (EDA)

Project Overview :
This project is part of my Data Science internship at Prodigy InfoTech. The objective is to perform data cleaning and exploratory data analysis (EDA) on the Titanic dataset. The analysis focuses on uncovering patterns and trends related to the survival of passengers based on various features.

Datasets Used :
train.csv: Contains the training data with features such as age, sex, passenger class, and whether the passenger survived.
test.csv: Contains the test data for which the survival status needs to be predicted.
gender_submission.csv: A sample submission file that indicates the correct format for the final predictions.

Project Structure :
EDA.py: The main Python script that performs data cleaning and EDA.
cleaned_train.csv: The cleaned version of the training dataset after handling missing values and unnecessary columns.
Key Steps


1. Data Cleaning
Handling Missing Values:
Filled missing values in the 'Age' column with the median age.
Filled missing values in the 'Embarked' column with the mode.
Dropped the 'Cabin' column due to a high percentage of missing values.

Data Type Conversion:
Converted categorical features like 'Sex' and 'Embarked' into numerical values for analysis.

2. Exploratory Data Analysis (EDA)
Univariate Analysis:
Visualized the distribution of survival status, age, and other key features.

Bivariate Analysis:
Analyzed survival rates based on sex and passenger class.
Examined the relationship between age and fare with respect to survival.
Correlation Analysis:

Generated a heatmap to visualize correlations between different features.
3. Findings
Survival Rates:
Higher survival rates were observed among females and passengers in higher classes (1st class).
Younger passengers and those who paid higher fares also had better chances of survival.

Data Visualization:
Used bar plots, scatter plots, and heatmaps to illustrate key insights from the data.

How to Run the Code
Clone the Repository:
git clone https://github.com/ChinmayDandekar7/titanic-eda.git

Install Required Libraries:
pip install pandas numpy matplotlib seaborn

Run the EDA Script:
python EDA.py

Results and Conclusion:
The project successfully demonstrated the importance of data cleaning and EDA in understanding and deriving meaningful insights from the Titanic dataset. The findings highlight key patterns in survival rates that could inform further predictive modeling efforts.

Acknowledgments :
The Titanic Dataset provided by Kaggle.

