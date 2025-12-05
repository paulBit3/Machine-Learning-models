"""
This model predicts us housing prices based on apartments bedrooms size 
"""
import pandas as pd
import seaborn as sb

#import MEAN

from sklearn.metrics import mean_absolute_error

#import LinearRegression
from sklearn.linear_model import LinearRegression

#import sklearn pipeline
from sklearn.pipeline import make_pipeline

#import simple imputer
from sklearn.impute import SimpleImputer

#preparing data by important our csv file
filepath = "your_csv_file_path_go_here"

#building the dataframe
df = pd.read_csv(filepath)

#check for the NaN(null values)
df.isnull().sum() / len(df)

#drop columns with NaN values
df.drop(columns=[
    "view",
    "sqft_basement"
], inplace = True)

##usig heatmap to multi collinearity issues

#correlation matrix
cor_matrix = df.select_dtypes(include ='float64').corr()

#convert correlation to a heatmap
sb.heatmap(cor_matrix)

##what we target here is housing prices and the features are bedroom size and bathroom

###Splitting Data

#target: price in USD
target = "price"

# y dependent variable
y_train = df[target]

#features: bedroom size, bathroom size
features = ["bedrooms", "bathrooms"]

# X as independent variable
X_train = df[features]

print(X_train)
print(y_train)

#### BUILDING OUR MODEL ####
""" In this implementation we will highlight the degree of relation(coefficient)
    between dependent and independent variables. We'll emphasis the MEAN ABSOLUTE ERROR
    We'll need to set specific standards to build a reliable model
"""

#let find the MEAN of our target(housing prices)
y_mean = y_train.mean()

#creating the prediction based on y_mean
y_pred_baseline = [y_mean] * len(y_train)

#calculating the MEAN ABSOLUTE ERROR
MAE_baseline = mean_absolute_error(y_train, y_pred_baseline)

#diplay the mean housing price and the baseline MAE
print("Mean housing price:", round(y_mean, 2))
print("Baseline MAE:", MAE_baseline)


## refining and improving our model

#let instantiate the lienear regression model
reg_model = LinearRegression()

#then we fit our model to the data
#this will run as a single predictor
reg_model.fit(X_train, y_train)
coef = reg_model.coef_[0]
#print(coef)

#Because we have a large dataset, then we
#create a pipline for our model adding simple imputer as transformer
# and linear regression as predictor
model = make_pipeline(
    SimpleImputer(),
    LinearRegression()
)

#then we fit our model to the data
model.fit(X_train, y_train)


##let do evaluation to see how our model perform against set standards

#performing house prediction
y_predictions = model.predict(X_train)

#calculate the MAE_pred_training
MAE_pred = mean_absolute_error(y_train, y_predictions)

#print the results
print("Training MAE: ", MAE_pred)

#prediction function
#y=mx+c

#We can calculate the simple coefficient of our model
coeff = round(reg_model.coef_[0], 2)

#calculate the coefficient of our model with transformer and/or pipeline
coeff = model.named_steps["linearregression"].coef_.round(2)

#We can calculate the simple intercept with our model
intercept = round(reg_model.intercept_, 2)

#calculate the intercept of our model with transformer
intercept = model.named_steps["linearregression"].intercept_.round(2)


#print our linear regression model

print("house price USD = {} + ({}* bedrooms) + ({}* bathrooms)"
    .format(intercept,
    coeff[0],
    coeff[1]
    )
)