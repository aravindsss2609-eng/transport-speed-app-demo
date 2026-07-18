import os
import time
from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit
import numpy as np

app = Flask(__name__)
app.config['SECRET_KEY'] = 'nexus-stream-telematics-2026'

try:
    import gevent
    import geventwebsocket
    chosen_async_mode = 'gevent'
    print("[+] Production WebSocket Core (gevent) successfully loaded.")
except ImportError:
    chosen_async_mode = None
    print("[*] gevent missing locally. Falling back to native development server thread mode.")

socketio = SocketIO(
    app, 
    cors_allowed_origins="*", 
    async_mode=chosen_async_mode, 
    websocket_ping_timeout=25, 
    websocket_ping_interval=5
)

device_registry = {}

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculates exact physical displacement across Earth surface in km using vectorized arrays."""
    R = 6371.0  
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)
    
    a = np.sin(delta_phi/2.0)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda/2.0)**2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return float(R * c)

def rule_based_classifier(speed, accel):
    """Robust physical inference framework for accurate multimodal transport classification."""
    abs_accel = abs(accel)
    if speed < 1.5:
        return "Stationary"
    elif speed <= 8.0:
        return "Walking / Jogging"
    elif speed <= 28.0:
        return "Biking / Eco-Mobility"
    elif speed <= 140.0:
        # Trains feature lower sustained structural acceleration updates than stop-and-go road cars
        return "Train" if abs_accel < 0.7 else "Car / Bus"
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
    print(f"[+] Active tracking pipeline opened for core context session: {session_id}")

@socketio.on('stop_calculation')
def handle_stop_calculation():
    session_id = request.sid
    if session_id in device_registry:
        device_registry[session_id]['active'] = False
        print(f"[-] Deactivated tracking session pipeline: {session_id}")

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
    
    try:
        lat = float(payload.get('lat'))
        lng = float(payload.get('lng'))
    except (TypeError, ValueError):
        return # Skip payload frame if coordinates are corrupted

    # Reliable dual-source time metric extraction layer
    raw_ts = payload.get('timestamp')
    ts = float(raw_ts) / 1000.0 if raw_ts else time.time()
    current_time = time.time()
    
    calculated_speed_kmh = 0.0
    incremental_dist = 0.0
    
    if state['last_lat'] is not None and state['last_lng'] is not None and state['last_ts'] is not None:
        incremental_dist = haversine_distance(state['last_lat'], state['last_lng'], lat, lng)
        time_delta_seconds = ts - state['last_ts']
        
        # Protect mathematical runtime against zero division errors from rapid browser updates
        if time_delta_seconds > 0.001:
            time_delta_hours = time_delta_seconds / 3600.0
            calculated_speed_kmh = incremental_dist / time_delta_hours
            
            # Universal Hardware GPS Fallback Validator: Use client-side computed telemetry array if backend values drop out
            client_speed = payload.get('speed')
            if (calculated_speed_kmh < 0.5 or calculated_speed_kmh > 250.0) and client_speed is not None:
                try:
                    # Convert incoming client speed metric directly to KM/H system
                    calculated_speed_kmh = float(client_speed) * 3.6
                except (TypeError, ValueError):
                    pass

    # Clean micro-movement sensor jitter filter
    if incremental_dist < 0.0005 and calculated_speed_kmh < 1.0:
        calculated_speed_kmh = 0.0
        incremental_dist = 0.0

    # Dynamic kinematic metrics processor
    acceleration_mps2 = 0.0
    g_force = 0.0
    
    if state['last_ts'] is not None:
        time_delta = ts - state['last_ts']
        if time_delta > 0.001:
            dv = (calculated_speed_kmh - state['last_speed']) / 3.6
            acceleration_mps2 = dv / time_delta
            g_force = acceleration_mps2 / 9.80665

    # Incremental Odometer execution mapping logic
    if calculated_speed_kmh > 0.5:
        state['total_distance_km'] += incremental_dist

    # Heuristic inference processing segment
    mode_prediction = rule_based_classifier(calculated_speed_kmh, acceleration_mps2)

    # Save tracking history state logs
    state['last_lat'] = lat
    state['last_lng'] = lng
    state['last_speed'] = calculated_speed_kmh
    state['last_ts'] = ts

    active_duration_sec = current_time - state['start_ts']
    
    emit('telemetry_processed', {
        'speed': round(max(0.0, calculated_speed_kmh), 1),
        'acceleration': round(acceleration_mps2, 2),
        'g_force': round(g_force, 2),
        'distance_travelled_km': round(state['total_distance_km'], 3),
        'duration_seconds': int(active_duration_sec),
        'predicted_mode': mode_prediction
    })

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    socketio.run(app, host='0.0.0.0', port=port, debug=True)