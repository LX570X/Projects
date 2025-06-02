# ===============================
# 🐍 PYTHON CHEAT SHEET
# ===============================

# === VARIABLES & DATA TYPES ===
x = 5               # Integer
pi = 3.14           # Float
name = "Alice"      # String
is_valid = True     # Boolean
nums = [1, 2, 3]    # List
person = {"name": "Bob", "age": 30}  # Dictionary
unique_vals = {1, 2, 3}  # Set
nothing = None      # Null

# === PRINTING ===
print("Hello, world!")
print(f"My name is {name} and I'm {x} years old.")

# === GETTING INPUT ===
# user_input = input("Enter something: ")
# print("You entered:", user_input)

# === TYPE CONVERSION ===
str(10)      # "10"
int("5")     # 5
float("3.14")# 3.14
bool(0)      # False

# === OPERATORS ===
# Arithmetic: + - * / // % **
# Comparison: == != > < >= <=
# Logical: and or not
# Assignment: = += -= *= /=
# Membership: in, not in

# === STRINGS ===
s = "Hello World"
print(s.lower())     # hello world
print(s.upper())     # HELLO WORLD
print(s[0])          # H
print(s[-1])         # d
print(s[0:5])        # Hello
print(len(s))        # 11

# === LISTS ===
fruits = ["apple", "banana", "cherry"]
fruits.append("orange")
fruits.remove("banana")
print(fruits[1])
print(fruits[-1])
print(len(fruits))

# === TUPLES ===
coords = (4, 5)
print(coords[0])

# === DICTIONARIES ===
person = {"name": "Alice", "age": 25}
print(person["name"])
person["age"] = 26
person["city"] = "Dubai"

# === CONDITIONALS ===
age = 20
if age >= 18:
    print("Adult")
elif age == 17:
    print("Almost adult")
else:
    print("Child")

# === LOOPS ===
# For loop
for i in range(5):
    print(i)

# While loop
count = 0
while count < 3:
    print("Counting:", count)
    count += 1

# === FUNCTIONS ===
def greet(name):
    return f"Hello, {name}!"

print(greet("Luna"))

# === CLASSES ===
class Dog:
    def __init__(self, name):
        self.name = name

    def bark(self):
        print(f"{self.name} says Woof!")

dog = Dog("Max")
dog.bark()

# === EXCEPTION HANDLING ===
try:
    val = int("abc")
except ValueError as e:
    print("Error:", e)
finally:
    print("Done")

# === FILE HANDLING ===
# Writing
with open("file.txt", "w") as f:
    f.write("Hello file!")

# Reading
with open("file.txt", "r") as f:
    print(f.read())

# === LIST COMPREHENSIONS ===
squares = [x**2 for x in range(5)]
print(squares)

# === LAMBDA FUNCTIONS ===
add = lambda a, b: a + b
print(add(2, 3))

# === IMPORTING MODULES ===
import math
print(math.sqrt(16))

from datetime import datetime
print(datetime.now())

# === JSON HANDLING ===
import json
data = {"name": "Alice", "age": 25}
json_str = json.dumps(data)
print(json_str)
data_back = json.loads(json_str)
print(data_back["name"])

# === VIRTUAL ENV (Terminal) ===
# python -m venv venv
# venv\Scripts\activate (Windows)
# source venv/bin/activate (Linux/macOS)

# === PIP COMMANDS (Terminal) ===
# pip install package
# pip uninstall package
# pip freeze > requirements.txt
# pip install -r requirements.txt

# === FLASK MINI EXAMPLE ===
# from flask import Flask
# app = Flask(__name__)

# @app.route('/')
# def home():
#     return "Hello API!"

# if __name__ == '__main__':
#     app.run(debug=True)

# -----------------------------
# 📌 PYTHON BASICS
# -----------------------------
# This section covers the foundational syntax of Python used in AI development.

x = 10                     # Integer value assigned to variable 'x'
name = "AI"                # String value assigned to variable 'name'
flag = True                # Boolean value indicating a true condition

def greet(name):          # Function definition with a parameter 'name'
    return f"Hello, {name}!"  # Return a formatted string using f-string

lst = [1, 2, 3]            # List (array-like) of integers
lst.append(4)              # Adds the number 4 to the end of the list

data = {"key": "value"}    # Dictionary with a key-value pair

for i in range(5):         # Loop that runs 5 times (from 0 to 4)
    print(i)               # Prints the current value of 'i'

if x > 5:                  # Conditional to check if x is greater than 5
    print("Large")         # Executes if condition is true

# -----------------------------
# 📌 NUMPY (NUMERICAL COMPUTING)
# -----------------------------
# NumPy is used for handling arrays, matrices, and linear algebra operations.

import numpy as np

a = np.array([[1, 2], [3, 4]])         # 2x2 matrix
b = np.eye(2)                          # 2x2 identity matrix
c = np.dot(a, b)                       # Matrix multiplication (dot product)
inv = np.linalg.inv(a)                # Matrix inverse
eigvals, eigvecs = np.linalg.eig(a)   # Eigenvalues and eigenvectors of matrix 'a'

# -----------------------------
# 📌 PANDAS (DATA HANDLING)
# -----------------------------
# Pandas is used for data manipulation and analysis using DataFrames.

import pandas as pd

df = pd.DataFrame({                   # Create a table-like structure
    "Name": ["Alice", "Bob"],
    "Score": [90, 85]
})
df.describe()                         # Shows statistical summary
df["Score"].mean()                    # Calculates the average score

# -----------------------------
# 📌 MATPLOTLIB (DATA VISUALIZATION)
# -----------------------------
# Matplotlib is used for plotting data and visualizing trends.

import matplotlib.pyplot as plt

x = np.linspace(0, 10, 100)           # Generate 100 points from 0 to 10
y = np.sin(x)                         # Apply sine function to each point
plt.plot(x, y)                        # Plot the x and y values
plt.title("Sine Wave")                # Set the title of the graph
plt.show()                            # Display the graph

# -----------------------------
# 📌 SCIKIT-LEARN (MACHINE LEARNING)
# -----------------------------
# Scikit-learn provides tools to build ML models like regression or classification.

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score

iris = load_iris()                   # Load built-in Iris dataset
X, y = iris.data, iris.target        # Feature matrix and target labels
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LinearRegression()           # Create a linear regression model
model.fit(X_train, y_train)          # Train it using training data
predictions = model.predict(X_test)  # Predict using test data

# Classification example:
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred) # Measure accuracy of predictions

# -----------------------------
# 📌 TENSORFLOW / KERAS (DEEP LEARNING)
# -----------------------------
# TensorFlow/Keras is used for building neural networks.

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential([
    Dense(10, activation="relu", input_shape=(4,)),  # Hidden layer
    Dense(3, activation="softmax")                   # Output layer for classification
])
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(X_train, y_train, epochs=10)              # Train the model for 10 iterations

# -----------------------------
# 📌 PYTORCH (ALTERNATIVE TO TENSORFLOW)
# -----------------------------
# PyTorch is widely used in research and for custom deep learning models.

import torch
import torch.nn as nn
import torch.optim as optim

X_tensor = torch.tensor(X_train, dtype=torch.float32)
y_tensor = torch.tensor(y_train, dtype=torch.long)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 10)
        self.fc2 = nn.Linear(10, 3)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.01)

for epoch in range(10):                 # Run for 10 training loops
    optimizer.zero_grad()              # Clear previous gradients
    outputs = net(X_tensor)            # Forward pass
    loss = criterion(outputs, y_tensor)# Compute loss
    loss.backward()                    # Backpropagation
    optimizer.step()                   # Update weights

# -----------------------------
# 📌 OPENCV (IMAGE PROCESSING)
# -----------------------------
# OpenCV is used to read, process, and analyze images.

import cv2
img = cv2.imread("image.jpg")                        # Load image
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)         # Convert to grayscale

# -----------------------------
# 📌 NLTK (NATURAL LANGUAGE PROCESSING)
# -----------------------------
# NLTK is used for basic text processing and tokenization.

import nltk
from nltk.tokenize import word_tokenize
nltk.download("punkt")                               # Download tokenizer data
tokens = word_tokenize("This is a test sentence.")   # Split into words

# -----------------------------
# 📌 SAVING & LOADING MODELS
# -----------------------------
# Save trained models to reuse them later without retraining.

import joblib
joblib.dump(model, "model.pkl")                      # Save model to file
loaded_model = joblib.load("model.pkl")              # Load model from file

# -----------------------------
# 📌 PIP & ENVIRONMENT TOOLS
# -----------------------------
# Useful terminal commands for managing packages and environments.

# Create virtual environment: python -m venv venv
# Activate on Windows:       venv\\Scripts\\activate
# Install package:           pip install numpy
# Save dependencies:         pip freeze > requirements.txt
# Install from file:         pip install -r requirements.txt

# -----------------------------
# 📌 FLASK MINI API EXAMPLE
# -----------------------------
# A tiny web API to deploy your AI models.

# from flask import Flask
# app = Flask(__name__)

# @app.route('/')
# def home():
#     return "Hello API!"

# if __name__ == '__main__':
#     app.run(debug=True)

# ============================================
# ✅ END OF PYTHON + AI CHEAT SHEET
# ============================================


# ============================================
# CNN EXAMPLE
# ============================================

# CNNs are used for image tasks like classification, detection, etc.

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist

# Load MNIST dataset (28x28 grayscale digits)
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Normalize and reshape input for CNN
X_train = X_train.reshape(-1, 28, 28, 1).astype("float32") / 255
X_test = X_test.reshape(-1, 28, 28, 1).astype("float32") / 255

# Define CNN model
cnn_model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),  # Feature extractor
    layers.MaxPooling2D((2,2)),                                          # Downsampling
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),                                                    # Flatten for dense layer
    layers.Dense(64, activation='relu'),                                 # Hidden layer
    layers.Dense(10, activation='softmax')                               # Output layer (10 classes)
])

cnn_model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

cnn_model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test))

# ============================================
# RNN EXAMPLE
# ============================================

# RNNs are useful for sequence data: text, time series, etc.

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
import numpy as np

# Sample text and character-level tokenization
text = "hello world"
chars = sorted(set(text))
char2idx = {c: i for i, c in enumerate(chars)}
idx2char = {i: c for c, i in char2idx.items()}
encoded = [char2idx[c] for c in text]

# Prepare sequences
X = []
y = []
seq_length = 3
for i in range(len(encoded) - seq_length):
    X.append(encoded[i:i+seq_length])
    y.append(encoded[i+seq_length])
X = np.array(X)
y = np.array(y)

# Define RNN model
rnn_model = Sequential([
    Embedding(input_dim=len(chars), output_dim=8, input_length=seq_length),
    SimpleRNN(16),
    Dense(len(chars), activation='softmax')
])

rnn_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')
rnn_model.fit(X, y, epochs=200, verbose=0)

# Predict next character after "hel"
input_seq = [char2idx[c] for c in "hel"]
input_seq = np.array(input_seq).reshape(1, -1)
pred_idx = np.argmax(rnn_model.predict(input_seq), axis=-1)[0]
print("Predicted next char:", idx2char[pred_idx])

