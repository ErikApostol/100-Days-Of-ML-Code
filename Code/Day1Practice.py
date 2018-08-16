# Step 1: import libraries
# The two must-imports: pandas and numpy
import pandas as pd # import and manage PANel DAta Sets
import numpy as np  # mathematical functions
from sklearn.preprocessing import Imputer, LabelEncoder, OneHotEncoder, StandardScalar
from sklearn.crossvalidation import train_test_split

# Step 2: import data
df = pd.read_csv("")
