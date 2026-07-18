import os
import time
from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit
import numpy as np

app = Flask(__name__)
app.config['SECRET_KEY'] = 'universal-telematics-2026'

# Optimized network socket layer for seamless handoffs between mobile towers
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet', websocket_ping_timeout=15, websocket_ping_interval=5)
device_registry = {}

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
        # Calculate raw physical displacement
        incremental_dist = haversine_distance(state['last_lat'], state['last_lng'], lat, lng)
        time_delta_seconds = ts - state['last_ts']
        
        if time_delta_seconds > 0.05: # High frequency compatibility layer
            time_delta_hours = time_delta_seconds / 3600.0
            calculated_speed_kmh = incremental_dist / time_delta_hours

    # --- THE DYNAMIC VELOCITY MULTI-PASS RESOLVER ---
    # Threshold 1: Noise Filtering (Stops values jumping when standing completely still)
    if incremental_dist < 0.00003 and calculated_speed_kmh < 0.5:
        calculated_speed_kmh = 0.0
    
    # Threshold 2: Low-Speed Walking / Jogging Mode
    elif calculated_speed_kmh > 0.5 and calculated_speed_kmh <= 6.0:
        # Allow raw calculation to pass through directly without filtering dampeners
        pass
        
    # Threshold 3: High-Speed Motorized Transport Mode (Trains, Cars, Metro)
    elif calculated_speed_kmh > 6.0:
        # Prevent temporary extreme GPS jumps (e.g. signal bouncing off a skyscraper)
        if calculated_speed_kmh > 250.0:
            calculated_speed_kmh = state['last_speed']

    # Physics vectors calculations
    acceleration_mps2 = 0.0
    g_force = 0.0
    
    if state['last_ts'] is not None:
        time_delta = ts - state['last_ts']
        if time_delta > 0.05:
            dv = (calculated_speed_kmh - state['last_speed']) / 3.6
            acceleration_mps2 = dv / time_delta
            g_force = acceleration_mps2 / 9.80665

    # Update Odometer metrics if actual movement is detected
    if calculated_speed_kmh > 0.0:
        state['total_distance_km'] += incremental_dist

    # Save state logs
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
        'duration_seconds': int(active_duration_sec)
    })

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    socketio.run(app, host='0.0.0.0', port=port)