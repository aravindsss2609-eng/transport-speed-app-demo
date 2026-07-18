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
    R = 6371.0  
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
    
    raw_speed = float(payload.get('speed', 0.0))
    lat = float(payload.get('lat', 0.0))
    lng = float(payload.get('lng', 0.0))
    ts = float(payload.get('timestamp', time.time() * 1000.0)) / 1000.0  

    current_time = time.time()
    
    filtered_speed = state['filter'].update(raw_speed)
    if filtered_speed < 0.5:  
        filtered_speed = 0.0

    acceleration_mps2 = 0.0
    g_force = 0.0
    
    if state['last_ts'] is not None:
        time_delta = ts - state['last_ts']
        if time_delta > 0.01:
            dv = (filtered_speed - state['last_speed']) / 3.6
            acceleration_mps2 = dv / time_delta
            g_force = acceleration_mps2 / 9.80665

    if state['last_lat'] is not None and state['last_lng'] is not None:
        incremental_dist = haversine_distance(state['last_lat'], state['last_lng'], lat, lng)
        if filtered_speed > 1.0:
            state['total_distance_km'] += incremental_dist

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
    # Render assigns ports dynamically via environment variables
    port = int(os.environ.get("PORT", 5000))
    socketio.run(app, host='0.0.0.0', port=port)