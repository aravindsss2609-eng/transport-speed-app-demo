import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

def generate_synthetic_telematics_dataset():
    """Generates an extensive telematics training dataset matching real physics."""
    print("[*] Generating production dataset...")
    np.random.seed(42)
    
    # UPGRADED Class definitions: True real-world physical bands
    # Profile: [Base Speed (km/h), Speed Variance, Acceleration Max (m/s²), Label]
    profiles = [
        (0.0, 0.1, 0.05, 'Stationary'), # Stationary is locked close to zero (pure device jitter)
        (4.5, 1.0, 0.75, 'Walking'),    # Pedestrian walking bounds
        (18.0, 5.0, 1.80, 'Biking'),    # Cycling speeds and agility
        (45.0, 15.0, 3.80, 'Car'),      # Urban and highway driving dynamics
        (85.0, 25.0, 1.10, 'Train')     # Consistent high speeds, very low acceleration curves
    ]
    
    data_list = []
    for base_speed, s_var, max_accel, label in profiles:
        samples = 3000 # Increased sampling to give the tree structure better edge definitions
        
        if label == 'Stationary':
            # Force absolute zero floor for static states, allowing tiny hardware drift deviations
            speeds = np.abs(np.random.normal(base_speed, s_var, samples))
        else:
            speeds = np.random.normal(base_speed, s_var, samples)
            
        speeds = np.clip(speeds, 0.0, 300.0) 
        accelerations = np.random.uniform(0.0, max_accel, samples)
        
        for i in range(samples):
            data_list.append({
                'Speed_Kmh': speeds[i],
                'Acceleration': accelerations[i],
                'Transport_Type': label
            })
            
    return pd.DataFrame(data_list)

def execute_pipeline():
    df = generate_synthetic_telematics_dataset()
    
    X = df[['Speed_Kmh', 'Acceleration']]
    y = df['Transport_Type']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("[*] Training Gradient Boosting Production Classifier...")
    # Tuned hyperparameters for rapid execution loops on micro-servers
    clf = GradientBoostingClassifier(
        n_estimators=120, 
        learning_rate=0.08, 
        max_depth=4, 
        random_state=42
    )
    clf.fit(X_train, y_train)
    
    # Validate accuracy
    predictions = clf.predict(X_test)
    print("\n--- Production Model Validation Report ---")
    print(classification_report(y_test, predictions))
    
    # Export clean artifact to backend
    joblib.dump(clf, 'transport_classifier.pkl')
    print("[+] Optimization Complete. 'transport_classifier.pkl' deployed successfully.")

if __name__ == "__main__":
    execute_pipeline()