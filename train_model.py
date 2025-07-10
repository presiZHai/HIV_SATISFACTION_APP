# ==========================
# train_model.py
# ==========================

import pandas as pd
import numpy as np
import os
import joblib
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from catboost import CatBoostClassifier

# Load environment variables
load_dotenv()

# Load data
data_path = "data/feature_engineered_data.csv"
df = pd.read_csv(data_path)
X = df.drop(columns=["Satisfaction"])
y = df["Satisfaction"]

# Encode categorical columns
categorical_cols = X.select_dtypes(include="object").columns.tolist()
encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
X[categorical_cols] = encoder.fit_transform(X[categorical_cols])

# Compute class weights
classes = np.unique(y)
class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y)
class_weight_dict = dict(zip(classes, class_weights))

# Label encode target
y_encoded = LabelEncoder().fit_transform(y)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, stratify=y_encoded, test_size=0.2, random_state=42)

# Define model and hyperparameter grid
model = CatBoostClassifier(verbose=0, random_state=42, class_weights=class_weights)
param_grid = {
    'depth': [4, 6, 8],
    'learning_rate': [0.01, 0.05, 0.1],
    'iterations': [100, 200]
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=cv, scoring='f1_weighted')
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
print("✅ Best parameters:", grid_search.best_params_)

# Save model and encoder
os.makedirs("model", exist_ok=True)
joblib.dump(best_model, "model/catboost_model.joblib")
joblib.dump(encoder, "model/encoder.joblib")
print("✅ Model and encoder saved.")