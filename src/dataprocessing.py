import pandas as pd

path = r"C:\Users\Jake King\python projects\Mouse-Dynamics-Challenge-master"

actions = pd.read_csv(path+r"/data/balabit_features_training.csv")
print(actions.head())
print(list(actions.columns))

# %%
# drop unnecessary features and convert  categorical data

actions = actions.join(pd.get_dummies(
    actions["type_of_action"], drop_first=True, prefix="action"))

actions = actions.join(pd.get_dummies(
    actions["direction_of_movement"], drop_first=True, prefix="direction"))

actions["user"] = actions["class"]

actions.drop(["class", "session", "n_from", "n_to", "type_of_action",
              "direction_of_movement"], axis=1, inplace=True)

actions.to_csv(path+r"\data\processed_balabit_data.csv", index=False)
