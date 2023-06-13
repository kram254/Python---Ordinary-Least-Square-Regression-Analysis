import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

# Loading the data
data = pd.read_csv('Income_dataset.csv')

# Selecting the relevant columns for the analysis
columns = ['SEX', 'MARITAL.STATUS', 'AGE', 'EDUCATION', 'OCCUPATION', 'INCOME']
data = data[columns]

# Handling the range values in the 'AGE' column if it contains string values
if data['AGE'].dtype == 'object':
    data['AGE'] = data['AGE'].apply(lambda x: float(x.split('-')[0]) if '-' in x else float(x))

# Preprocessing pipeline for handling missing values 
numeric_cols = ['AGE']
categorical_cols = ['SEX', 'MARITAL.STATUS', 'EDUCATION', 'OCCUPATION']

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean'))
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Defining the design matrix (independent variables) and the dependent variable
X = data.drop('INCOME', axis=1)
y = data['INCOME']

# Linear regression model using statsmodels
X = preprocessor.fit_transform(X)
X = sm.add_constant(X)  

# Adding a constant column for intercept
model = sm.OLS(y, X)
results = model.fit()

# Extracting the coefficients, p-values, and standard errors
coefficients = results.params[1:] 
p_values = results.pvalues[1:]
std_errors = results.bse[1:]

# Creating a results DataFrame
categorical_names = preprocessor.transformers_[1][1]['encoder'].get_feature_names_out(categorical_cols)
results_data = {
    'Variable': list(numeric_cols) + list(categorical_names),
    'Coefficient': list(coefficients),
    'P-value': list(p_values),
    'Standard Error': list(std_errors)
}
results_data['Variable'].append('Intercept')
results_data['Coefficient'].append(results.params[0])
results_data['P-value'].append(results.pvalues[0])
results_data['Standard Error'].append(results.bse[0])
results = pd.DataFrame(results_data)

# Printing the regression results table
print(results)