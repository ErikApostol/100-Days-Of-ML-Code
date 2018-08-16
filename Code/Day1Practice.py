# Step 1: import libraries
# The two must-imports: pandas and numpy
import pandas as pd # import and manage PANel DAta Sets
import numpy as np  # mathematical functions
from sklearn.preprocessing import Imputer, LabelEncoder, OneHotEncoder, StandardScalar
from sklearn.crossvalidation import train_test_split

# Step 2: import data
df = pd.read_csv("Data.csv") # Do not take any line as header
X_arr = df.iloc[:, :-1].values # .iloc returns DataFrame, .values returns ndarray
Y_arr = df.iloc[:, -1].values

# Step 3: impute some values to cells with missing data
imputer = Imputer(missing_values='NaN', strategy='median', axis=0)
X_arr = imputer.fit_transform(X_arr)

# Step 4: Turn categorical variables into numbers
# LabelEncoder
labelencoder = LabelEncoder()
Y_arr = labelencoder.fit_transform(Y_arr)
# OneHotEncoder
# sparse : boolean, default=True. Will return sparse matrix if set True, else will return an array.
onehotencoder = OneHotEncoder(categorical_variables=[0], sparse=False) 
Y_arr = onehotencoder.fit_transform(Y_arr)

# Step 5: trans_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X_arr, Y_arr, test_size=0.2)

# Step 6: Normalize input
standardscalar = StandardScalar()
X_train = standardscalar.fit_transform(X_train)
X_test = standardscalar.fit_transform(X_test)