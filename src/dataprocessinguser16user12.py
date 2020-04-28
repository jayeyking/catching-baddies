import pandas as pd

path = r"C:\Users\Jake King\python projects\Mouse-Dynamics-Challenge-master"

actions = pd.read_csv("../data/balabit_features_training.csv")
print(actions.head())
print(list(actions.columns))

# %%
# create new target variable and remove class imbalance in new dataframe
# bal_actions
user16_data = actions[actions["class"] == 16].sample(n=7655)
user12_data = actions[actions["class"] == 12]

bal_actions = user16_data.append(user12_data).reset_index(drop=True)

# %%
# drop unnecessary features and convert categorical data
bal_actions = bal_actions.join(pd.get_dummies(
    bal_actions["type_of_action"], drop_first=True, prefix="action"))
bal_actions = bal_actions.join(pd.get_dummies(
    bal_actions["class"], drop_first=True, prefix="user"))
bal_actions.drop(["session", "n_from", "n_to", "type_of_action"],
                 axis=1, inplace=True)

bal_actions.to_csv(path+r"\data\user16_vs_user12.csv", index=False)
