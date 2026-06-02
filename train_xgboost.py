import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import os

print("⚡ Démarrage de l'entraînement de XGBoost...")

# 1. Charger les données préparées à la Phase 2
train_df = pd.read_csv('data/features_train.csv')
X_train = train_df.drop(columns=['target'])
y_train = train_df['target']

# Utiliser les dernières lignes pour la validation rapide
X_test = X_train.tail(2000)
y_test = y_train.tail(2000)

# 2. Initialiser le modèle XGBoost
# n_jobs=-1 permet d'utiliser tous les cœurs de ton processeur pour aller plus vite
xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42, n_jobs=-1)

# 3. Entraîner le modèle
print("Entraînement en cours (XGBoost)...")
xgb_model.fit(X_train, y_train)
print("Modèle XGBoost entraîné avec succès !")

# 4. Évaluation
predictions = xgb_model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("\n--- RÉSULTATS XGBOOST ---")
print(f"Erreur Absolue Moyenne (MAE) : {mae:.2f} km")
print(f"Score R² (Précision globale) : {r2 * 100:.2f} %")
print("-------------------------\n")

# 5. Sauvegarde du modèle
os.makedirs('models', exist_ok=True)
model_path = 'models/xgboost_model.pkl'
joblib.dump(xgb_model, model_path)
print(f"Modèle sauvegardé dans : {model_path}")