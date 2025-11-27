import pandas as pd

#importing train test split to train our model
from sklearn.model_selection import train_test_split

#importing standardization
from sklearn.preprocessing import StandardScaler



#print(pd.__version__)

#creating dataframe
df = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/data/master/delaney_solubility_with_descriptors.csv')
print(df)

#data pre-processing using X and y from df
X, y = df.iloc[:, :-1].values, df.iloc[:, -1].values

print(X, y)

#data splitting(we use Scikit-Learn )
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

print(X_train, X_test, y_train, y_test)

#scalling feature
st_desc = StandardScaler()
X_train_std = st_desc.fit_transform(X_train)
X_test_std = st_desc.transform(X_test)

sc_lot = pd.DataFrame(X_train_std).describe()
print(sc_lot)

