import os
import time
from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit
import numpy as np

app = Flask(__name__)
app.config['SECRET_KEY'] = 'nexus-stream-telematics-2026'

# Enforce strict WebSocket mode for low-latency delivery over cellular mobile towers
# Change async_mode to 'gevent' for compatibility
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='gevent', websocket_ping_timeout=15, websocket_ping_interval=5)
device_registry = {}

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculates exact physical displacement across Earth surface in km."""
    R = 6371.0  
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)
    
    a = np.sin(delta_phi/2.0)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda/2.0)**2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return R * c

def rule_based_classifier(speed, accel):
    """Instant physical inference framework for multimodal classification."""
    abs_accel = abs(accel)
    if speed < 0.8:
        return "Stationary"
    elif speed <= 7.0:
        return "Walking / Jogging"
    elif speed <= 25.0:
        return "Biking / Eco-Mobility"
    elif speed <= 130.0:
        # Trains accelerate very smoothly compared to stop-and-go road traffic
        return "Train" if abs_accel < 0.6 else "Car / Bus"
    else:
        return "High-Speed Transit"

@app.route('/')
def dashboard():
    return render_template('index.html')

@socketio.on('start_calculation')
def handle_start_calculation():
    session_id = request.sid
    device_registry[session_id] = {
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
    
    lat = float(payload.get('lat', 0.0))
    lng = float(payload.get('lng', 0.0))
    ts = float(payload.get('timestamp', time.time() * 1000.0)) / 1000.0  
    current_time = time.time()
    
    calculated_speed_kmh = 0.0
    incremental_dist = 0.0
    
    if state['last_lat'] is not None and state['last_lng'] is not None and state['last_ts'] is not None:
        incremental_dist = haversine_distance(state['last_lat'], state['last_lng'], lat, lng)
        time_delta_seconds = ts - state['last_ts']
        
        if time_delta_seconds > 0.05:
            time_delta_hours = time_delta_seconds / 3600.0
            calculated_speed_kmh = incremental_dist / time_delta_hours

    # Spatial Window Jitter Filter: Kills cellular tower location bounce when standing dead still
    if incremental_dist < 0.00002 and calculated_speed_kmh < 0.4:
        calculated_speed_kmh = 0.0
        incremental_dist = 0.0

    # Physics vectors calculations
    acceleration_mps2 = 0.0
    g_force = 0.0
    
    if state['last_ts'] is not None:
        time_delta = ts - state['last_ts']
        if time_delta > 0.05:
            # Convert delta speed from km/h back into m/s for kinematic correctness
            dv = (calculated_speed_kmh - state['last_speed']) / 3.6
            acceleration_mps2 = dv / time_delta
            g_force = acceleration_mps2 / 9.80665

    # Odometer processing logic
    if calculated_speed_kmh > 0.0:
        state['total_distance_km'] += incremental_dist

    # Fetch heuristic inference classification based on real-world kinematics
    mode_prediction = rule_based_classifier(calculated_speed_kmh, acceleration_mps2)

    # Save tracking history state logs
    state['last_lat'] = lat
    state['last_lng'] = lng
    state['last_speed'] = calculated_speed_kmh
    state['last_ts'] = ts

    active_duration_sec = current_time - state['start_ts']
    
    emit('telemetry_processed', {
        'speed': round(calculated_speed_kmh, 1),
        'acceleration': round(acceleration_mps2, 2),
        'g_force': round(g_force, 2),
        'distance_travelled_km': round(state['total_distance_km'], 3),
        'duration_seconds': int(active_duration_sec),
        'predicted_mode': mode_prediction
    })

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    socketio.run(app, host='0.0.0.0', port=port)