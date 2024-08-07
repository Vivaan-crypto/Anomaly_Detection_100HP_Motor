import os
import pickle

import matplotlib.pyplot as plt
import pandas as pd
from flask import Flask, redirect, render_template, request, url_for

# Configuring Flask
app = Flask(__name__)
os.makedirs("uploads", exist_ok=True)
file_path = None

# Load the trained model
model_path = "models/Anomaly_Detection_100HP_Motor.pkl"
with open(model_path, "rb") as file:
    clf = pickle.load(file)


# Define routes
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_file():
    file = request.files["data_file"]

    if file:
        global file_path
        filename = file.filename
        file_path = os.path.join("uploads", filename)
        file.save(file_path)
        return redirect(url_for("predict"))
    else:
        print("NO FILE!!!")
        return render_template("index.html")


@app.route("/predict/<filename>")
def predict():
    data = pd.read_csv(file_path)

    # Assuming the data has the same structure as the training data
    y_data = data.pop("label")
    x_data = data

    y_pred = clf.predict(x_data)
    acc = accuracy(y_pred, y_data)

    # Plot the results
    plot(test_df=data, y_pred=y_pred)
    plot_url = url_for("static", filename="plot.png")
    return render_template("result.html", plot_url=plot_url, acc=acc)


# Helper Functions
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


def plot(test_df, y_pred):
    # -------Testing-------#
    # Plotting
    plt.figure(figsize=(11, 6))
    idxs = []
    for i in range(0, len(y_pred)):
        if y_pred[i] == 1:
            idxs.append(i)

    # Plot all in one graph
    plt.plot(test_df["temperature"], label="Temperature(F)", marker="d", color="purple")
    plt.plot(test_df["current"], label="Current(A)", marker="d", color="orange")
    plt.plot(test_df["vibration"], label="Vibration(mm/s)", marker="d", color="green")
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
    plt.tight_layout()  # Adjust layout to make room for the legend
    plt.savefig("static/plot.png", bbox_inches="tight")  # Save with tight bounding box
    plt.close()


app.run(host="0.0.0.0", port=80)
