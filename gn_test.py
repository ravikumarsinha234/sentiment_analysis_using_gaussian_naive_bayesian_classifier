import pandas as pd
import numpy as np
import pickle as pkl
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

TEST_PATH = "test_gn.csv"
CONFUSION_PATH = "confusion_gn.png"
PLOT_PATH = "confidence_gn.png"
PARAMETERS_PATH = "parameters_gn.pkl"

## Your code here
print(f"Read parameters from {PARAMETERS_PATH}")
attribute_name = np.array(pkl.load(open(PARAMETERS_PATH, "rb"))[0])
prior_lst = np.array(pkl.load(open(PARAMETERS_PATH, "rb"))[1])
array_mean = np.array(pkl.load(open(PARAMETERS_PATH, "rb"))[2])
array_std = np.array(pkl.load(open(PARAMETERS_PATH, "rb"))[3])

# Read the test data
test_df = pd.read_csv(TEST_PATH)
# print(test_df.head)
X = test_df.iloc[:, :-1].to_numpy(dtype=np.float64)
n = X.shape[0]
Y_df = test_df.iloc[:, -1]
print(
    f'Can expect {prior_lst[0]:.1f}% accuracy by guessing "{attribute_name[0]}" every time.'
)
print(f"Read {len(test_df)} rows from {TEST_PATH}")

# log of prior_lst
log_prior_lst = np.log(prior_lst)

# log of likelihood
log_likelihood = np.zeros((n, len(attribute_name)))
for i in range(n):
    for j in range(len(attribute_name)):
        log_likelihood[i, j] = np.sum(
            np.log(1 / (np.sqrt(2 * np.pi) * array_std[j]))
            - ((X[i] - array_mean[j]) ** 2) / (2 * array_std[j] ** 2)
        )

# log of posterior
log_posterior = log_likelihood + log_prior_lst

# predict
predict = np.argmax(log_posterior, axis=1)
# print(predict)

# Calculate the prediction probabilities for each of the label using predict and log_posterior
prediction_probs_array = (
    np.exp(log_posterior - np.max(log_posterior, axis=1)[:, np.newaxis])
    / np.sum(
        np.exp(log_posterior - np.max(log_posterior, axis=1)[:, np.newaxis]), axis=1
    )[:, np.newaxis]
)

print("Here are 10 rows of results:")
for k in range(10):
    print(f"\tGT={Y_df.iloc[k]}->", end="")
    for j in range(len(attribute_name)):
        print(
            f"\t{attribute_name[j]}: {prediction_probs_array[k, j]*100.0:.1f}%", end=""
        )
    print()

prediction_Y = attribute_name[np.argmax(log_posterior, axis=1)]

print("\n*** Analysis ***")

no_of_correct = np.sum(prediction_Y == Y_df)
accuracy = no_of_correct / n

print(
    f"{n} data points analyzed, {no_of_correct} correct ({accuracy * 100.0:.1f}% accuracy)"
)

# confusion matrix
confusion = confusion_matrix(Y_df, prediction_Y, labels=attribute_name)
print("Confusion:\n", confusion)

# Save out a confusion matrix plot
fig, ax = plt.subplots()
cm_display = ConfusionMatrixDisplay(
    confusion_matrix=confusion, display_labels=attribute_name
)
cm_display.plot(ax=ax, cmap="Blues", colorbar=False)
fig.savefig("confusion_gn.png")
print("Wrote confusion matrix plot to confusion_gn.png")

print("\n*** Making a plot ****")

confidence = np.max(prediction_probs_array, axis=1)
steps = 32
thresholds = np.linspace(0.2, 1.0, steps)
correct_ratio = np.zeros(steps)
confident_ratio = np.zeros(steps)

for i in range(steps):
    threshold = thresholds[i]
    if np.sum(confidence > threshold) != 0:
        correct_ratio[i] = np.sum(
            (confidence > threshold) & (prediction_Y == Y_df)
        ) / np.sum(confidence > threshold)
    else:
        correct_ratio[i] = 1
    confident_ratio[i] = np.sum(confidence >= threshold) / n

fig, ax = plt.subplots()
ax.set_title("Confidence and Accuracy Are Correlated")
ax.set_xlabel("Confidence Threshold")
ax.yaxis.set_major_formatter(lambda x, pos: f"{x*100.0:.0f}%")
ax.hlines(
    prior_lst[np.argmax(prior_lst)],
    0.2,
    1,
    "blue",
    linestyle="dashed",
    linewidth=0.8,
    label=f"Accuracy Guessing {attribute_name[np.argmax(prior_lst)]}",
)
ax.plot(
    thresholds, correct_ratio, "blue", linewidth=0.8, label="Accuracy Above Threshod"
)
ax.plot(
    thresholds,
    confident_ratio,
    "r",
    linestyle="dashed",
    linewidth=0.8,
    label="Test data scoring above threshold",
)
ax.legend()
fig.savefig("confidence_gn.png")
print(f'Saved to "{PLOT_PATH}".')
