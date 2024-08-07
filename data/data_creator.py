import numpy as np
import pandas as pd

# Set random seed for reproducibility
np.random.seed(60)

# Define sizes
train_data_size = 100
anomalies_data_size = 10
test_normal_size = 250
test_anomalies_size = 75

# Generate random normal data for training set
train_normal_data = {
    "temperature": np.random.normal(
        loc=np.random.uniform(60, 80),
        scale=np.random.uniform(4, 6),
        size=train_data_size,
    ),
    "current": np.random.normal(
        loc=np.random.uniform(8, 12),
        scale=np.random.uniform(0.5, 1.5),
        size=train_data_size,
    ),
    "voltage": np.random.normal(
        loc=np.random.uniform(200, 240),
        scale=np.random.uniform(5, 15),
        size=train_data_size,
    ),
    "vibration": np.random.normal(
        loc=np.random.uniform(0.25, 0.35),
        scale=np.random.uniform(0.03, 0.07),
        size=train_data_size,
    ),
}

# Generate random anomaly data for training set
train_anomalies_data = {
    "temperature": np.random.normal(
        loc=np.random.uniform(90, 110),
        scale=np.random.uniform(3, 7),
        size=anomalies_data_size,
    ),
    "current": np.random.normal(
        loc=np.random.uniform(15, 25),
        scale=np.random.uniform(1, 3),
        size=anomalies_data_size,
    ),
    "voltage": np.random.normal(
        loc=np.random.uniform(160, 200),
        scale=np.random.uniform(10, 30),
        size=anomalies_data_size,
    ),
    "vibration": np.random.normal(
        loc=np.random.uniform(0.7, 0.9),
        scale=np.random.uniform(0.05, 0.15),
        size=anomalies_data_size,
    ),
}

# Convert to DataFrame
train_normal_df = pd.DataFrame(train_normal_data).round(2)
train_anomalies_df = pd.DataFrame(train_anomalies_data).round(2)

# Create labels
train_normal_labels = np.zeros(train_data_size)
train_anomalies_labels = np.ones(anomalies_data_size)

# Add labels to DataFrame
train_normal_df["label"] = train_normal_labels
train_anomalies_df["label"] = train_anomalies_labels

# Combine normal and anomaly data for training set
train_df = pd.concat([train_normal_df, train_anomalies_df], ignore_index=True)

# Shuffle the training set
train_df = train_df.sample(frac=1).reset_index(drop=True)

# Generate random normal data for testing set
np.random.seed(10000)
test_normal_data = {
    "temperature": np.random.normal(
        loc=np.random.uniform(60, 80),
        scale=np.random.uniform(4, 6),
        size=test_normal_size,
    ),
    "current": np.random.normal(
        loc=np.random.uniform(8, 12),
        scale=np.random.uniform(0.5, 1.5),
        size=test_normal_size,
    ),
    "voltage": np.random.normal(
        loc=np.random.uniform(200, 240),
        scale=np.random.uniform(5, 15),
        size=test_normal_size,
    ),
    "vibration": np.random.normal(
        loc=np.random.uniform(0.25, 0.35),
        scale=np.random.uniform(0.03, 0.07),
        size=test_normal_size,
    ),
}

# Generate random anomaly data for testing set
test_anomalies_data = {
    "temperature": np.random.normal(
        loc=np.random.uniform(90, 110),
        scale=np.random.uniform(3, 7),
        size=test_anomalies_size,
    ),
    "current": np.random.normal(
        loc=np.random.uniform(15, 25),
        scale=np.random.uniform(1, 3),
        size=test_anomalies_size,
    ),
    "voltage": np.random.normal(
        loc=np.random.uniform(160, 200),
        scale=np.random.uniform(10, 30),
        size=test_anomalies_size,
    ),
    "vibration": np.random.normal(
        loc=np.random.uniform(0.7, 0.9),
        scale=np.random.uniform(0.05, 0.15),
        size=test_anomalies_size,
    ),
}

# Convert to DataFrame
test_normal_df = pd.DataFrame(test_normal_data).round(2)
test_anomalies_df = pd.DataFrame(test_anomalies_data).round(2)

# Create labels
test_normal_labels = np.zeros(test_normal_size)
test_anomalies_labels = np.ones(test_anomalies_size)

# Add labels to DataFrame
test_normal_df["label"] = test_normal_labels
test_anomalies_df["label"] = test_anomalies_labels

# Combine normal and anomaly data for test set
test_df = pd.concat([test_normal_df, test_anomalies_df], ignore_index=True)

# Shuffle the test set
test_df = test_df.sample(frac=1).reset_index(drop=True)


# Converting to CSV
train_df.to_csv("../data/motor_data_train.csv", index=False)
test_df.to_csv("../data/motor_data_test.csv", index=False)


# Display the results
print("Training DataFrame:")
print(train_df.head())

print("\nTesting DataFrame:")
print(test_df.head())
