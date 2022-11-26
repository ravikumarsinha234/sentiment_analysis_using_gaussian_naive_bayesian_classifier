import pandas as pd
import numpy as np
import pickle as pkl

TRAIN_PATH = "train_gn.csv"
PARAMETERS_PATH = "parameters_gn.pkl"


def show_array(category_label, array, labels):
    print(f"\t{category_label} -> ", end="")
    for i in range(len(array)):
        print(f"{labels[i]}:{array[i]: >7.4f}     ", end="")
    print()


train_df = pd.read_csv(TRAIN_PATH)
X_df = train_df.iloc[:, :-1]
Y_df = train_df.iloc[:, -1]
n = len(X_df)
d = len(X_df.columns)
print(f"Read {n} samples with {d} attributes from {TRAIN_PATH}")

## Your code here
attribute_name = Y_df.unique()
attribute_name.sort()
print("Priors:")
prior_lst = []
for i in range(len(attribute_name)):
    prior_lst.append(train_df["Y"].value_counts()[i] / n)
    print(f"\t{attribute_name[i]}: {train_df['Y'].value_counts()[i]/n*100:.1f}%")

"""compute the mean and standard deviation of each attribute for each class"""
pt_mean = train_df.pivot_table(index="Y", aggfunc="mean")
pt_std = train_df.pivot_table(index="Y", aggfunc="std")
# print out the mean and std of each attribute for each class
for i in range(len(attribute_name)):
    print(f"{attribute_name[i]}:")
    show_array("Means", pt_mean.iloc[i], pt_mean.columns)
    show_array("Stdvs", pt_std.iloc[i], pt_std.columns)

means = np.zeros((len(attribute_name), d))
stds = np.zeros((len(attribute_name), d))
for i, word in enumerate(attribute_name):
    means[i] = X_df[Y_df == word].mean()
    stds[i] = X_df[Y_df == word].std()
with open("parameters_gn.pkl", "wb") as f:
    pkl.dump((attribute_name, prior_lst, means, stds), f)

# prior of train_df code
print(f"Wrote parameters to {PARAMETERS_PATH}")
