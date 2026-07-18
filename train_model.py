import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

def generate_synthetic_telematics_dataset():
    """Generates an extensive telematics training dataset."""
    print("[*] Generating production dataset...")
    np.random.seed(42)
    
    # Class definitions: [Base Speed, Speed Variance, Acceleration Max, Label]
    profiles = [
        (1.5, 0.5, 0.2, 'Stationary'),
        (5.0, 1.2, 0.6, 'Walking'),
        (18.0, 4.0, 1.5, 'Biking'),
        (55.0, 20.0, 3.5, 'Car'),
        (110.0, 25.0, 1.2, 'Train')
    ]
    
    data_list = []
    for base_speed, s_var, max_accel, label in profiles:
        # Generate 2000 telemetry packets for each mode
        samples = 2000
        speeds = np.random.normal(base_speed, s_var, samples)
        speeds = np.clip(speeds, 0, 300) # clip to real world speed limits
        
        accelerations = np.random.uniform(0, max_accel, samples)
        
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
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("[*] Training Gradient Boosting Production Classifier...")
    # Using Gradient Boosting for rapid inference pipelines
    clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
    clf.fit(X_train, y_train)
    
    # Validate accuracy
    predictions = clf.predict(X_test)
    print("\n--- Production Model Validation Report ---")
    print(classification_report(y_test, predictions))
    
    # Export clean artifact to backend
    joblib.dump(clf, 'transport_classifier.pkl')
    print("[+] Optimization Complete. 'transport_classifier.pkl' deployed.")

if __name__ == "__main__":
    execute_pipeline()