import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import normalize
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

actions = pd.read_csv("../data/balabit_features_training.csv")
print(actions.head())
print(list(actions.columns))

# %%
# create a "user16" variable for the new target variable
actions["user16"] = [1 if i == 16 else 0 for i in actions["class"]]

# %%
# remove class imbalance and create new balanced dataframe bal_actions
user16_data = actions[actions["user16"] == 1]
not_user16 = actions[actions["user16"] == 0].sample(n=10767)

bal_actions = user16_data.append(not_user16).reset_index(drop=True)

# %%
# drop unnecessary features and convert  categorical data
bal_actions = bal_actions.join(pd.get_dummies(
    bal_actions["type_of_action"], drop_first=True, prefix="action"))
bal_actions.drop(["class", "session", "n_from", "n_to", "type_of_action"],
                 axis=1, inplace=True)

# %%
# split into features and target variable
X = bal_actions.drop("user16", axis=1)
y = bal_actions["user16"]

norm_X = normalize(X)

X_train, X_test, y_train, y_test = train_test_split(norm_X, y, test_size=0.3,
                                                    random_state=42)

# %%
# knn algo testing 1 to 50 neighbors to see which number of neighbors is best

scores = []

for i in range(1, 50):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    scores.append(knn.score(X_test, y_test))

# %%
# plot scores for different numbers of neighbors

fig, ax = plt.subplots(figsize=(16, 8))
sns.lineplot(x=list(range(1, 50)), y=scores, ax=ax, lw=3)
ax.set_title("Searching for the Plateau", color="black", size=20, y=1.03)
ax.set_xlabel("Number of Neighbors", color="black", size=20)
ax.set_ylabel("Score", color="black", size=20)
ax.tick_params(colors="black")

ax.annotate("Plateau here",
            xy=(0.35, 0.65),
            xytext=(0.55, 0.4),
            xycoords="figure fraction",
            textcoords="figure fraction",
            size=20,
            arrowprops=dict(arrowstyle='->',
                            connectionstyle='arc3',
                            color='grey',
                            lw=3))

plt.show()

# %% knn suggests 12 neighbors might be best
