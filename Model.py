import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, StandardScaler as SparkScaler, VectorAssembler
from pyspark.ml import Pipeline
import streamlit as st


spark = SparkSession.builder \
    .appName("AirQualityPrediction") \
    .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
    .config("spark.network.timeout", "600s") \
    .config("spark.execpyutor.heartbeatInterval", "60s") \
    .getOrCreate()


file_path = r'C:\Users\rdroh\OneDrive\Desktop\Project\air_quality_data.csv'
spark_df = spark.read.csv(file_path, header=True, inferSchema=True)

# Data Preprocessing in Spark using Pipeline
spark_df = spark_df.dropna().dropDuplicates()

indexer = StringIndexer(inputCol="Air Quality", outputCol="label")
assembler = VectorAssembler(inputCols=['Temperature', 'Humidity', 'Wind Speed'], outputCol='features')
pipeline = Pipeline(stages=[indexer, assembler])
pipeline_model = pipeline.fit(spark_df)
spark_df = pipeline_model.transform(spark_df)

scaler = SparkScaler(inputCol='features', outputCol='scaled_features')
scaler_model = scaler.fit(spark_df)
spark_df = scaler_model.transform(spark_df)

# Sample the data to reduce size
sample_df = spark_df.sample(False, 0.1)

# Convert Spark DataFrame to Pandas DataFrame
pandas_df = sample_df.toPandas()

# Encode categorical labels using LabelEncoder
label_encoder = LabelEncoder()
pandas_df['Air Quality'] = label_encoder.fit_transform(pandas_df['Air Quality'])

# Split data into features and labels
X = np.array(pandas_df['scaled_features'].tolist())
y = pandas_df['Air Quality'].values

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define a custom dataset
class AirQualityDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# Create DataLoaders
train_dataset = AirQualityDataset(X_train, y_train)
test_dataset = AirQualityDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Model Definition
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(3, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, len(label_encoder.classes_))

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = SimpleNN()

# Training the Model
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.7)

class EarlyStopping:
    def __init__(self, patience=3, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.best_loss = np.Inf
        self.counter = 0
        self.should_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
            if self.verbose:
                print(f"Validation loss decreased: {val_loss:.4f}")
        else:
            self.counter += 1
            if self.verbose:
                print(f"Validation loss did not improve: {val_loss:.4f}")
            if self.counter >= self.patience:
                self.should_stop = True

early_stopping = EarlyStopping(patience=3, verbose=True)

def train_model(model, train_loader, criterion, optimizer, scheduler, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}')

        scheduler.step()


        early_stopping(epoch_loss)
        if early_stopping.should_stop:
            print("Early stopping")
            break

train_model(model, train_loader, criterion, optimizer, scheduler)


def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_labels.extend(labels.numpy())
            all_preds.extend(predicted.numpy())

    accuracy = 100 * correct / total
    report = classification_report(all_labels, all_preds, target_names=label_encoder.classes_, zero_division=0)
    return accuracy, report

test_accuracy, test_report = evaluate_model(model, test_loader)


def predict(model, new_data):
    model.eval()
    with torch.no_grad():
        new_data_tensor = torch.tensor(new_data, dtype=torch.float32)
        outputs = model(new_data_tensor)
        _, predictions = torch.max(outputs, 1)
        return predictions.numpy()


st.title("Air Quality Prediction")

st.write("## Model Performance")
st.write(f"Test Accuracy: {test_accuracy:.2f}%")
st.text_area("Classification Report", test_report)

st.write("## Predict Air Quality")


temperature = st.number_input("Enter Temperature", min_value=-50.0, max_value=50.0, value=25.0)
humidity = st.number_input("Enter Humidity", min_value=0.0, max_value=100.0, value=50.0)
wind_speed = st.number_input("Enter Wind Speed", min_value=0.0, max_value=150.0, value=10.0)


if st.button("Predict"):

    new_data_df = pd.DataFrame([[temperature, humidity, wind_speed]], columns=['Temperature', 'Humidity', 'Wind Speed'])
    new_data_spark_df = spark.createDataFrame(new_data_df)


    new_data_spark_df = pipeline_model.transform(new_data_spark_df)
    new_data_spark_df = scaler_model.transform(new_data_spark_df)

    new_data_pandas = new_data_spark_df.toPandas()
    new_data_features = np.array(new_data_pandas['scaled_features'].tolist())

    prediction = predict(model, new_data_features)
    st.write(f"Predicted Air Quality Index: {label_encoder.inverse_transform(prediction)[0]}")




spark.stop()
st.balloons()
st.snow()