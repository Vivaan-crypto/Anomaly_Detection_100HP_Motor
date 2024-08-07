import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from flask import Flask, render_template
from sklearn.ensemble import RandomForestClassifier

# Configuring Flask
app = Flask(__name__)

train_path = "../data/motor_data_train.csv"
test_path = "../data/motor_data_test.csv"

# Getting the datasets
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

# Splliting x and y in training datasets
y_train = train_df.pop("label")
x_train = train_df

# Splitting x and y in testing datasets
y_test = test_df.pop("label")
x_test = test_df


def accuracy(y_pred, y_test):
    correct_preds = 0
    wrong_preds_idxs = []
    for i in range(0, len(y_pred)):
        if y_pred[i] == y_test[i]:
            correct_preds += 1
        else:
            wrong_preds_idxs.append(i)

    acc = round(correct_preds / len(y_test), 4) * 100
    return float(acc)


# train_df = pd.read_csv(args.train_file)


clf = RandomForestClassifier(n_estimators=100, max_depth=25)

# Fitting the training dataset
clf.fit(x_train, y_train)

# Making predictions
y_pred = clf.predict(x_test)

# Finding the accuracy of the model
accuracy(y_pred, y_test)
# -----------------------GRAPHING THE DATA--------------------------#


def plot(train_df, test_df, y_pred, plot_train, plot_test):
    if plot_train:
        # -------Training-------#
        # Plotting
        plt.figure(figsize=(10, 6))

        # Plot all in one graph
        plt.plot(
            train_df["temperature"], label="Temperature(F)", marker="d", color="purple"
        )
        plt.plot(train_df["current"], label="Current(A)", marker="d", color="orange")
        plt.plot(
            train_df["vibration"], label="Vibration(mm/s)", marker="d", color="green"
        )
        plt.plot(train_df["voltage"], label="Voltage(V)", marker="d", color="blue")

        plt.title("Sensor Readings")
        plt.xlabel("Seconds")
        plt.ylabel("Values")
        plt.legend()

        plt.show()
    if plot_test:
        # -------Testing-------#

        # Plotting
        plt.figure(figsize=(10, 6))
        idxs = []
        for i in range(0, len(y_pred)):
            if y_pred[i] == 1:
                idxs.append(i)

        # Plot all in one graph
        plt.plot(
            test_df["temperature"], label="Temperature(F)", marker="d", color="purple"
        )
        plt.plot(test_df["current"], label="Current(A)", marker="d", color="orange")
        plt.plot(
            test_df["vibration"], label="Vibration(mm/s)", marker="d", color="green"
        )
        plt.plot(test_df["voltage"], label="Voltage(V)", marker="d", color="blue")

        # Highlight specific points with red outlines
        plt.scatter(
            idxs,
            test_df.loc[idxs, "temperature"],
            edgecolor="red",
            facecolor="red",
            s=100,
            label=None,
        )
        plt.scatter(
            idxs,
            test_df.loc[idxs, "current"],
            edgecolor="red",
            facecolor="red",
            s=100,
            label=None,
        )
        plt.scatter(
            idxs,
            test_df.loc[idxs, "vibration"],
            edgecolor="red",
            facecolor="red",
            s=100,
            label=None,
        )
        plt.scatter(
            idxs,
            test_df.loc[idxs, "voltage"],
            edgecolor="red",
            facecolor="red",
            s=100,
            label=None,
        )

        plt.title("Sensor Readings")
        plt.xlabel("Seconds")
        plt.ylabel("Values")
        plt.legend(
            loc="upper left",
            bbox_to_anchor=(1, 1),
        )

        plt.show()


plot(
    train_df=train_df, test_df=test_df, y_pred=y_pred, plot_train=False, plot_test=True
)
# Export model
with open("models/Anomaly_Detection_100HP_Motor.pkl", "wb") as file:
    pickle.dump(clf, file)
