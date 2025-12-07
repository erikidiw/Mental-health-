import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from category_encoders.target_encoder import TargetEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import numpy as np
import sys

# ==========================
# 🔧 Load Dataset
# ==========================
try:
    df = pd.read_csv("student_depression_dataset.csv")
    print("Data 'student_depression_dataset.csv' berhasil dimuat.")
except FileNotFoundError:
    print("Error: student_depression_dataset.csv tidak ditemukan. Pastikan file ada di direktori yang sama.")
    sys.exit(1)
except Exception as e:
    print(f"Error saat memuat data: {e}")
    sys.exit(1)

# ==========================
# 🔧 Custom Preprocessing Classes
# ==========================

# 1. Custom Mapper untuk Ordinal Mapping
class CustomOrdinalMapper:
    def __init__(self, mappings):
        # Mappings: List of tuples (col_name, {map_dict})
        self.mappings = {col: map_dict for col, map_dict in mappings}
        self.cols = [col for col, _ in mappings]
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        X_copy = X.copy()
        for col, mapping in self.mappings.items():
            if col in X_copy.columns:
                # Map, fillna 0, dan pastikan float
                # Ini juga mengatasi '?' yang sudah diganti '0' di langkah cleaning
                X_copy[col] = X_copy[col].map(mapping).fillna(0).astype(float)
        # Hanya kembalikan kolom yang dipetakan
        return X_copy[self.cols]

# ==========================
# 🔧 Preprocessing Definitions
# ==========================

# 1. Cleaning function (Menghilangkan quotes/spasi dan mengganti '?' di Financial Stress)
def clean_data(df):
    df_copy = df.copy()
    for col in ['Sleep Duration', 'Financial Stress']:
        if col in df_copy.columns:
            df_copy[col] = df_copy[col].astype(str).str.replace("'", "").str.strip()
            
    # Mengganti '?' di Financial Stress dengan '0' (berdasarkan skrip asli Anda)
    if 'Financial Stress' in df_copy.columns:
        df_copy['Financial Stress'] = df_copy['Financial Stress'].replace('?', '0')
    
    return df_copy

# Bersihkan Data
df_cleaned = clean_data(df)

# Drop unused/unnecessary columns
df_cleaned = df_cleaned.drop(columns=['id', 'Work Pressure', 'Job Satisfaction'], errors='ignore')

# Definisikan X dan y
X = df_cleaned.drop(columns=['Depression'])
y = df_cleaned['Depression']

# Definisikan mapping dan kolom
ordinal_mapping = [
    ('Sleep Duration', {'Less than 5 hours': 1, '5-6 hours': 2, '7-8 hours': 3, 'More than 8 hours': 4, 'Others': 0}),
    ('Financial Stress', {'1.0': 1, '2.0': 2, '3.0': 3, '4.0': 4, '5.0': 5, '0': 0}),
    ('Have you ever had suicidal thoughts ?', {'No': 0, 'Yes': 1}),
    ('Family History of Mental Illness', {'No': 0, 'Yes': 1}),
]

label_cols = ['Gender', 'Dietary Habits', 'Degree', 'Social Weakness']
target_cols = ['City', 'Profession']
ordinal_cols_names = [col for col, _ in ordinal_mapping]

# Kolom numerik yang tersisa
numerical_cols = [col for col in X.columns if col not in ordinal_cols_names + label_cols + target_cols]

# ==========================
# 🚀 Training All Artifacts
# ==========================

X_processed = X.copy()

# 1. Ordinal Mapping
custom_mapper = CustomOrdinalMapper(ordinal_mapping)
X_processed[ordinal_cols_names] = custom_mapper.fit_transform(X_processed)

# 2. Label Encoding
le_encoders = {}
for col in label_cols:
    le = LabelEncoder()
    # Pastikan data dikonversi ke string sebelum fit
    X_processed[col] = le.fit_transform(X_processed[col].astype(str))
    le_encoders[col] = le

# 3. Target Encoding
te = TargetEncoder(min_samples_leaf=20, smoothing=10)
X_processed[target_cols] = te.fit_transform(X_processed[target_cols], y)

# 4. Final Data Check (Pastikan semua float)
X_processed = X_processed.apply(pd.to_numeric, errors='coerce').fillna(0).astype(float)


# 5. Pipeline (Scaling dan Model)
pipeline = Pipeline(steps=[
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(n_estimators=200, random_state=42))
])

# Fit Pipeline hanya dengan data yang sudah di-encode
pipeline.fit(X_processed, y)

# ==========================
# 💾 Save Pipeline and Encoders (Artifacts)
# ==========================

artifacts = {
    'pipeline': pipeline,
    'label_encoders': le_encoders,
    'target_encoder': te,
    'ordinal_mapper': custom_mapper,
    'feature_cols': X.columns.tolist(),
}

joblib.dump(artifacts, 'pipeline_artifacts.pkl')

print("\n-------------------------------------------------")
print("✅ File 'pipeline_artifacts.pkl' berhasil dibuat.")
print("   Sekarang Anda bisa menjalankan 'streamlit run app.py'.")
print("-------------------------------------------------")
