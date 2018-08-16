# Step 1: import libraries
# The two must-imports: pandas and numpy
import pandas as pd # import and manage PANel DAta Sets
import numpy as np  # mathematical functions
from sklearn.preprocessing import Imputer, LabelEncoder, OneHotEncoder, StandardScalar
from sklearn.crossvalidation import train_test_split

# Step 2: import data
df = pd.read_csv("pima-indians-diabetes.data.csv", header=None) # Do not take any line as header
X_arr = df.iloc[:, :-1].values # .iloc returns DataFrame, .values returns ndarray
Y_arr = df.iloc[:, -1].values

# Step 3: impute some values to cells with missing data
