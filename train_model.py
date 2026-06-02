import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import os

print("Démarrage de la Phase 3 : Entraînement de l'IA...")

# 1. Charger le dataset préparé à la Phase 2
# Pour cet exemple de base, nous chargeons le fichier d'entraînement créé
train_df = pd.read_csv('data/features_train.csv')

# Séparer les caractéristiques (X) et la cible à prédire (y)
X_train = train_df.drop(columns=['target'])
y_train = train_df['target']

# Créer un faux jeu de test pour l'exemple (en situation réelle, on chargerait un fichier features_test.csv)
# Pour valider rapidement notre code, on prend les 2000 dernières lignes de l'entraînement
X_test = X_train.tail(2000)
y_test = y_train.tail(2000)

# 2. Initialiser le modèle Random Forest
print("Entraînement du modèle Random Forest (cela peut prendre quelques secondes)...")
model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)

# 3. Entraîner l'IA
model.fit(X_train, y_train)
print("Modèle entraîné avec succès !")

# 4. Évaluation du modèle
print("Évaluation des performances...")
predictions = model.predict(X_test)

# Calcul des métriques
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("\n--- RÉSULTATS DU MODÈLE ---")
print(f"Erreur Absolue Moyenne (MAE) : {mae:.2f} km")
print(f"Score R² (Précision globale) : {r2 * 100:.2f} %")
print("---------------------------\n")

# 5. Sauvegarder l'IA sur le disque
# Cela permettra au Dashboard (Phase 4) de réutiliser cette IA sans réentraîner
os.makedirs('models', exist_ok=True)
model_path = 'models/random_forest_model.pkl'
joblib.dump(model, model_path)
print(f"Modèle sauvegardé avec succès dans : {model_path}")