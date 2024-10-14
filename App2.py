import tkinter as tk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

warnings.filterwarnings("ignore", message="X does not have valid feature names")

# Read Dataset
df = pd.read_csv(r"E:\Office Project work\Vehicle Recomendation system\Dataset1.csv")


# find Outlier
import pandas as pd
import numpy as np
from scipy import stats

df = pd.DataFrame(df)

# Calculate Z-scores for each column
z_scores = np.abs(stats.zscore(df))

# Set threshold and remove rows where any Z-score is above threshold
threshold = 3
df = df[(z_scores < threshold).all(axis=1)]

# Split data
X = df[['Engine rpm', 'Lub oil pressure', 'Fuel pressure', 'Coolant pressure', 'lub oil temp', 'Coolant temp']]
y = df['Engine Condition']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Define the model
model = RandomForestClassifier(random_state=42)

# Define the parameter grid
param_grid = {
    'n_estimators': [10, 20, 30],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None]
}

# Perform Grid Search
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best parameters and model
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

# Tkinter UI
def submit():
    inputs = [float(entry.get()) for entry in entries.values()]
    input_array = np.array(inputs).reshape(1, -1)
    prediction = best_model.predict(input_array)
    # Display the prediction
    if prediction[0] == 1:
        result_label.config(text=f"Engine Condition is Good.")
    else:
        result_label.config(text=f"Engine Condition is Bad.")

root = tk.Tk()
root.title("Engine Parameters Input")

parameters = [
    "Engine RPM",
    "Lub Oil Pressure",
    "Fuel Pressure",
    "Coolant Pressure",
    "Lub Oil Temperature",
    "Coolant Temperature"
]

entries = {}
for param in parameters:
    label = tk.Label(root, text=f"Enter {param}:")
    label.pack(pady=5)
    entry = tk.Entry(root, width=30)
    entry.pack(pady=5)
    entries[param] = entry

submit_button = tk.Button(root, text="Submit", command=submit)
submit_button.pack(pady=10)

result_label = tk.Label(root, text="")
result_label.pack(pady=10)

root.mainloop()
