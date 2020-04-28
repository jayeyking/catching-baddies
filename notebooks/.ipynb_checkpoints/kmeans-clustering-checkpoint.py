import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
# from sklearn.pipeline import Pipeline

actions = pd.read_csv("../data/balabit_features_training.csv")
print(actions.head())
print(list(actions.columns))

# %%
# simplify the data set and drop the categorical features
means = [col for col in actions.columns if "mean" in col]
X = actions[means]

# %%
# to demonstrate need for scaling look at a pairplot
plt.figure(figsize=(16, 8))
sns.pairplot(X)

plt.show()

# %%
# scale the data and check again
norm_X = normalize(X)

plt.figure(figsize=(16, 8))
sns.pairplot(pd.DataFrame(norm_X))

plt.show()

# %%
# check the elbow plot to see if a suggested number of clusters comes out
clusters = []

for i in range(1, 15):
    km = KMeans(n_clusters=i).fit(X)
    clusters.append(km.inertia_)

# %%
fig, ax = plt.subplots(figsize=(16, 8))
sns.lineplot(x=list(range(1, 15)), y=clusters, ax=ax, lw=3)
ax.set_title("Searching for the Elbow", color="black", size=20, y=1.03)
ax.set_xlabel("Clusters", color="black", size=20)
ax.set_ylabel("Inertia", color="black", size=20)
ax.tick_params(colors="black")

ax.annotate("Possible elbow point",
            xy=(0.25, 0.37),
            xytext=(0.35, 0.6),
            xycoords="figure fraction",
            textcoords="figure fraction",
            size=20,
            arrowprops=dict(arrowstyle='->',
                            connectionstyle='arc3',
                            color='grey',
                            lw=3))

ax.annotate("Possible elbow point",
            xy=(0.3, 0.3),
            xytext=(0.46, 0.52),
            xycoords="figure fraction",
            textcoords="figure fraction",
            size=20,
            arrowprops=dict(arrowstyle='->',
                            connectionstyle='arc3',
                            color='grey',
                            lw=3))

plt.show()

# %%
km3 = KMeans(n_clusters=3).fit(X)
X["cluster_labels_km3"] = km3.labels_

plt.figure(figsize=(16, 8))
sns.scatterplot(X["mean_curv"], X["mean_jerk"], hue=X["cluster_labels_km3"],
                palette=sns.color_palette('hls', 3))
plt.title("KMeans with 3 Clusters", color="black", size=20, y=1.03)
plt.xlabel("Mean Curveature", color="black", size=20)
plt.ylabel("Mean Jerk", color="black", size=20)
plt.xticks(color="black")
plt.yticks(color="black")

plt.show()

# %%
km4 = KMeans(n_clusters=4).fit(X)
X["cluster_labels_km4"] = km4.labels_

plt.figure(figsize=(16, 8))
sns.scatterplot(X["mean_curv"], X["mean_jerk"], hue=X["cluster_labels_km4"],
                palette=sns.color_palette('hls', 4))
plt.title("KMeans with 4 Clusters", color="black", size=20, y=1.03)
plt.xlabel("Mean Curveature", color="black", size=20)
plt.ylabel("Mean Jerk", color="black", size=20)
plt.xticks(color="black")
plt.yticks(color="black")

plt.show()

# %%
# 3 clusters seems to work but logical as algo is probably just picking up
# action type.