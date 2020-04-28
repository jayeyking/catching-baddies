import pandas as pd

from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split

# %%

actions = pd.read_csv("../data/processed_balabit_actions.csv")
print(actions.head())
print(list(actions.columns))

# %%
# split into features and target variable
X = actions.drop("user16", axis=1)
y = actions["user16"]

norm_X = normalize(X)

X_train, X_test, y_train, y_test = train_test_split(norm_X, y, test_size=0.3,
                                                    random_state=42)
