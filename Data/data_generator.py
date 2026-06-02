import numpy as np
import pandas as pd

def generate_trip(profile='eco', duration_s=1800, dt=1.0):
    """
    Génère un trajet synthétique réaliste selon un profil de conduite.
    """
    # Configuration des profils de conduite (Vitesse max, accélération, mode, récupération)
    params = {
        'eco':    {'v_max': 70,  'throttle_mean': 15, 'agility': 0, 'recup': 2},
        'normal': {'v_max': 100, 'throttle_mean': 30, 'agility': 1, 'recup': 1},
        'sport':  {'v_max': 130, 'throttle_mean': 55, 'agility': 2, 'recup': 0}
    }
    
    p = params[profile]
    n = int(duration_s / dt)
    t = np.arange(n) * dt
    
    # Simulation de la vitesse (courbe sinusoïdale + bruit aléatoire)
    speed = np.clip(p['v_max'] * np.sin(t / 200) + np.random.normal(0, 8, n), 0, p['v_max'])
    
    # Calcul des pédales (accélérateur et frein) à partir de l'accélération (dérivée de la vitesse)
    dv = np.gradient(speed)
    throttle = np.clip(dv * 3 + p['throttle_mean'] + np.random.normal(0, 5, n), -30, 100)
    brake = np.clip(-dv * 5 + np.random.normal(0, 3, n), 0, 100)
    
    # Modèle physique simplifié de la consommation (kWh/100km)
    consumption = 10 + speed * 0.08 + np.abs(throttle) * 0.1
    if p['recup'] == 2:
        consumption -= np.clip(-dv * 2, 0, 3) # Récupération d'énergie au freinage
        
    # Simulation de la décharge de la batterie (Capacité max fixe : 60 kWh)
    energy_max = 60.0
    energy_consumed = np.cumsum(consumption * (speed / 3600 / 100) * dt)
    remaining_energy = np.clip(energy_max - energy_consumed, 0, energy_max)
    
    # Calcul mathématique de l'Autonomie restante réelle (en km) -> Notre cible pour l'IA
    remaining_dist = np.where(consumption > 0, remaining_energy / (consumption / 100), 0)
    
    # Structuration finale sous forme de tableau (DataFrame Pandas) avec les signaux officiels du DBC
    df = pd.DataFrame({
        'timestamp_s': t,
        'profile': profile,
        'sigVehicleSpeed': speed,
        'sigThrottle': throttle,
        'sigBrakeForce': brake,
        'sigMotorEfficiency': np.clip(85 - np.abs(throttle) * 0.1, 60, 98),
        'sigEnergyPer100km': consumption,
        'sigRemainingEnergy': remaining_energy,
        'sigAgility': p['agility'],
        'sigRecuperationLevel': p['recup'],
        'sigRemainingDistance': remaining_dist  # Variable cible à prédire
    })
    return df

# --- Génération d'une base de données multi-trajets ---
all_trips = []
for profile in ['eco', 'normal', 'sport']:
    for duration in [1800, 3600, 5400]:  # Génère des trajets de 30min, 1h, et 1h30
        trip = generate_trip(profile=profile, duration_s=duration)
        all_trips.append(trip)

# Fusionner tous les trajets générés et sauvegarder dans un fichier CSV
dataset = pd.concat(all_trips, ignore_index=True)
dataset.to_csv('data/synthetic_dataset.csv', index=False)
print(f"Félicitations ! Dataset créé avec {len(dataset)} lignes.")