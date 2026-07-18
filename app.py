import os
import time
from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit
import numpy as np

app = Flask(__name__)
app.config['SECRET_KEY'] = 'keyless-telematics-system-2026'

socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')
device_registry = {}

class TelematicsKalmanFilter:
    def __init__(self, initial_value=0.0):
        self.X = initial_value  
        self.P = 1.0            
        self.Q = 0.05           
        self.R = 0.8            

    def update(self, measurement):
        self.P = self.P + self.Q
        kalman_gain = self.P / (self.P + self.R)
        self.X = self.X + kalman_gain * (measurement - self.X)
        self.P = (1 - kalman_gain) * self.P
        return self.X

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371.0  # Earth's radius in kilometers
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)
    
    a = np.sin(delta_phi/2.0)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda/2.0)**2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return R * c

@app.route('/')
def dashboard():
    return render_template('index.html')

@socketio.on('start_calculation')
def handle_start_calculation():
    session_id = request.sid
    device_registry[session_id] = {
        'filter': TelematicsKalmanFilter(),
        'last_lat': None,
        'last_lng': None,
        'last_speed': 0.0,
        'last_ts': None,
        'total_distance_km': 0.0,
        'start_ts': time.time(),
        'active': True
    }

@socketio.on('stop_calculation')
def handle_stop_calculation():
    session_id = request.sid
    if session_id in device_registry:
        device_registry[session_id]['active'] = False

@socketio.on('disconnect')
def handle_device_disconnection():
    device_registry.pop(request.sid, None)

@socketio.on('telemetry_ingress')
def process_telemetry_stream(payload):
    session_id = request.sid
    if session_id not in device_registry:
        return

    state = device_registry[session_id]
    if not state.get('active', False):
        return
    
    raw_speed = payload.get('speed')
    lat = float(payload.get('lat', 0.0))
    lng = float(payload.get('lng', 0.0))
    ts = float(payload.get('timestamp', time.time() * 1000.0)) / 1000.0  

    current_time = time.time()
    
    # --- UPGRADE: MATH-BASED BACKUP SPEED FALLBACK ---
    calculated_speed_kmh = 0.0
    incremental_dist = 0.0
    
    if state['last_lat'] is not None and state['last_lng'] is not None and state['last_ts'] is not None:
        incremental_dist = haversine_distance(state['last_lat'], state['last_lng'], lat, lng)
        time_delta_hours = (ts - state['last_ts']) / 3600.0
        
        # Calculate speed mathematically if the time delta is sane (> 0.5 seconds)
        if time_delta_hours > 0.000138: 
            calculated_speed_kmh = incremental_dist / time_delta_hours

    # Enforce hardware speed if available and valid; otherwise use mathematical derivation
    if raw_speed is not None and float(raw_speed) > 0.0:
        actual_speed_input = float(raw_speed) * 3.6  # Convert m/s to km/h if native metric
    else:
        actual_speed_input = calculated_speed_kmh

    # Pass the optimized speed tracking vector through the Kalman Filter
    filtered_speed = state['filter'].update(actual_speed_input)
    
    # Filter out static GPS jitter standing still at the station
    if filtered_speed < 1.5 or (incremental_dist * 1000 < 0.5):  
        filtered_speed = 0.0

    # Calculate real-time acceleration and G-force matrix
    acceleration_mps2 = 0.0
    g_force = 0.0
    
    if state['last_ts'] is not None:
        time_delta = ts - state['last_ts']
        if time_delta > 0.1:
            dv = (filtered_speed - state['last_speed']) / 3.6
            acceleration_mps2 = dv / time_delta
            g_force = acceleration_mps2 / 9.80665

    # Update Odometer metrics if the vehicle is visibly moving
    if filtered_speed > 0.0 and incremental_dist > 0.0:
        # Prevent massive anomalies caused by sudden GPS location jumps
        if filtered_speed < 200.0: 
            state['total_distance_km'] += incremental_dist

    # Save state vectors for the next sequence iteration
    state['last_lat'] = lat
    state['last_lng'] = lng
    state['last_speed'] = filtered_speed
    state['last_ts'] = ts

    active_duration_sec = current_time - state['start_ts']
    
    emit('telemetry_processed', {
        'speed': round(filtered_speed, 1),
        'acceleration': round(acceleration_mps2, 2),
        'g_force': round(g_force, 2),
        'distance_travelled_km': round(state['total_distance_km'], 3),
        'duration_seconds': int(active_duration_sec)
    })

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    socketio.run(app, host='0.0.0.0', port=port)