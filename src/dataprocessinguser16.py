import pandas as pd

path = r"C:\Users\Jake King\python projects\Mouse-Dynamics-Challenge-master"

actions = pd.read_csv("../data/balabit_features_training.csv")
print(actions.head())
print(list(actions.columns))

# %%
# create new target variable and remove class imbalance in new dataframe
# bal_actions
user16_data = actions[actions["class"] == 16]
not_user16 = actions[actions["class"] != 16].sample(n=10767)

bal_actions = user16_data.append(not_user16).reset_index(drop=True)

# %%
# drop unnecessary features and convert  categorical data
bal_actions = bal_actions.join(pd.get_dummies(
    bal_actions["type_of_action"], drop_first=True, prefix="action"))
bal_actions = bal_actions.join(pd.get_dummies(
    bal_actions["direction_of_movement"], drop_first=True, prefix="direction"))
bal_actions["user_16"] = [1 if i == 16 else 0 for i in bal_actions["class"]]
bal_actions.drop(["class", "session", "n_from", "n_to", "type_of_action",
                 "direction_of_movement"],
                 axis=1, inplace=True)

bal_actions.to_csv(path+r"\data\user16_vs_the_world.csv", index=False)
