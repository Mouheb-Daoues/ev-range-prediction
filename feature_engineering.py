import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

print("Démarrage du Feature Engineering...")

# 1. Charger le dataset généré à la Phase 1
df = pd.read_csv('data/synthetic_dataset.csv')

# 2. Définir la fenêtre glissante (30 secondes)
# Comme notre simulation génère 1 ligne par seconde, 30 lignes = 30 secondes
window = 30

# 3. Création des caractéristiques temporelles (Features)
print("Calcul des moyennes glissantes sur 30 secondes...")
df['speed_mean_30s'] = df['sigVehicleSpeed'].rolling(window=window, min_periods=1).mean()
df['speed_std_30s'] = df['sigVehicleSpeed'].rolling(window=window, min_periods=1).std().fillna(0)

df['consumption_mean_30s'] = df['sigEnergyPer100km'].rolling(window=window, min_periods=1).mean()

# Indicateur de nervosité : écart-type de la position de la pédale d'accélérateur
df['throttle_std_30s'] = df['sigThrottle'].rolling(window=window, min_periods=1).std().fillna(0)

# 4. Nettoyage : Supprimer les colonnes inutiles ou redondantes pour l'IA
# On garde les nouvelles features calculées et les paramètres fixes
features_columns = [
    'speed_mean_30s', 'speed_std_30s', 'consumption_mean_30s', 'throttle_std_30s',
    'sigRemainingEnergy', 'sigAgility', 'sigRecuperationLevel'
]
target_column = 'sigRemainingDistance'

X = df[features_columns]
y = df[target_column]

# 5. Découpage chronologique (Train / Validation / Test)
# Contrairement à d'autres projets, sur des trajets (séries temporelles), on ne mélange pas au hasard.
# On garde la fin des trajets pour tester l'IA.
print("Découpage des données (70% Entraînement, 15% Validation, 15% Test)...")
total_len = len(df)
train_end = int(total_len * 0.70)
val_end = int(total_len * 0.85)

X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
X_val, y_val = X.iloc[train_end:val_end], y.iloc[train_end:val_end]
X_test, y_test = X.iloc[val_end:], y.iloc[val_end:]

# 6. Normalisation des données (Standardisation)
# L'IA a du mal quand une donnée vaut 130 (vitesse) et une autre vaut 0.8 (énergie).
# On met tout à la même échelle (moyenne = 0, écart-type = 1).
print("Normalisation des données avec StandardScaler...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# 7. Sauvegarde des datasets préparés pour l'IA
# Pour que ce soit propre, on reconstruit des DataFrames pour les sauvegarder en CSV
train_dataset = pd.DataFrame(X_train_scaled, columns=features_columns)
train_dataset['target'] = y_train.values

train_dataset.to_csv('data/features_train.csv', index=False)
print("Félicitations ! Fichier 'features_train.csv' créé avec succès dans data/.")
print(f"Nombre de caractéristiques préparées : {len(features_columns)}")