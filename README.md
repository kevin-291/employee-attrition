# IBM HR Analytics: Employee Attrition & Performance Analysis

This repository contains a Python code that performs a comprehensive analysis of employee attrition and performance data using various data visualization techniques and machine learning models. The code is designed to provide insights into factors influencing employee attrition and to develop predictive models for identifying employees at risk of leaving the organization.

## Dataset

The analysis is based on the `IBM Employee Attrition & Performance Analysis` dataset, which contains information about employees, including their age, income, job role, department, education, and attrition status.

## Analysis Steps

1. **Data Loading and Exploration**: The code begins by importing the necessary libraries and loading the dataset into a pandas DataFrame. It performs initial data exploration by checking the data types, null values, and unique values in categorical and numerical columns. Irrelevant columns are dropped from the dataset.

2. **Data Visualization**: The code employs several visualization techniques using Plotly and Seaborn to explore the relationships between different variables and employee attrition. It examines factors such as age distribution, monthly income, job satisfaction, department, job role, salary hikes, work experience diversity, and education levels.

3. **Data Preprocessing**: The code handles categorical variables by converting them to numeric using label encoding and one-hot encoding. Correlation analysis is performed to identify highly correlated features. Ordinal encoding is applied to categorical variables, and numerical features are dropped. Normalization is performed using MinMaxScaler, and the data is split into training and testing sets. SMOTE (Synthetic Minority Over-sampling Technique) is applied to handle class imbalance in the target variable (Attrition).

4. **Model Development**: The code trains and evaluates several machine learning models for employee attrition prediction, including Logistic Regression, Random Forest Classifier, Gradient Boosting Classifier, AdaBoost Classifier, and Gaussian Process Classifier. For each model, the code computes the accuracy score, plots the confusion matrix, and displays the classification report.


5. **Key Findings and Recommendations**: The code presents key findings from the analysis, including insights into gender disparity, age dynamics, income levels, job satisfaction, departmental differences, job role impact, salary increment influence, educational background, and the effects of salary, stock options, and work-life balance on employee attrition. Based on the findings, the code provides recommendations for improving employee retention and organizational success.

## Requirements

Install the required dependencies listed in requirements.txt:

```bash
pip install -r requirements.txt
```

## Usage

1. Clone or download this repository to your local machine.

```bash
git clone https://github.com/kevin-291/employee-attrition.git
```

2. Navigate to the project directory.

```bash
cd employee-attrition
```

3. Open the Jupyter Notebook or Python script containing the code.

4. Run the code cells or script to perform the analysis and generate the visualizations and model outputs.

## Contributing

Contributions to this project are welcome. If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.



