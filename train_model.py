import pandas as pd
import numpy as np
import os
import joblib
from dotenv import load_dotenv

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

from catboost import CatBoostClassifier

# Load environment variables
load_dotenv()

# Load data
data_path = "data/feature_engineered_data.csv"
if not os.path.exists(data_path):
    raise FileNotFoundError(f"❌ File not found: {data_path}")

df = pd.read_csv(data_path)
if "Satisfaction" not in df.columns:
    raise ValueError("❌ 'Satisfaction' column not found in data")

X = df.drop(columns=["Satisfaction"])
y = df["Satisfaction"]

# Identify categorical features
categorical_cols = X.select_dtypes(include="object").columns.tolist()

# Label encode target
y_encoder = LabelEncoder()
y_encoded = y_encoder.fit_transform(y)
joblib.dump(y_encoder, "model/label_encoder.joblib")

# Compute class weights
classes = np.unique(y_encoded)
class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_encoded)
class_weight_dict = dict(zip(classes, class_weights))

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, stratify=y_encoded, test_size=0.2, random_state=42
)

# Hyperparameter tuning using CatBoost with categorical support
model = CatBoostClassifier(
    verbose=0,
    random_state=42,
    class_weights=class_weights
)

param_grid = {
    'depth': [4, 6, 8],
    'learning_rate': [0.01, 0.05, 0.1],
    'iterations': [100, 200]
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=cv,
    scoring='f1_weighted'
)
grid_search.fit(X_train, y_train, cat_features=categorical_cols)

best_model = grid_search.best_estimator_
print("✅ Best parameters:", grid_search.best_params_)

# Feature selection
top_n = 10
importances = best_model.feature_importances_
top_features_selected = X.columns[np.argsort(importances)[::-1][:top_n]].tolist()
print(f"✅ Top {top_n} features: {top_features_selected}")

# Subset top features
X_top = X[top_features_selected]
top_categorical = [col for col in top_features_selected if col in categorical_cols]

# Train final model on top features
final_model = CatBoostClassifier(verbose=0, random_state=42, class_weights=class_weights)
final_model.fit(X_top, y_encoded, cat_features=top_categorical)

# Save everything
os.makedirs("model", exist_ok=True)
joblib.dump(final_model, "model/top10_model.joblib")
joblib.dump(top_features_selected, "model/important_features.joblib")
joblib.dump(top_categorical, "model/top_categorical_features.joblib")  # Optional

# Save categories for frontend dropdowns
categories = {
    col: sorted(df[col].dropna().unique().tolist())
    for col in top_categorical
}
joblib.dump(categories, "model/categories.joblib")

print("✅ Final model trained and saved.")