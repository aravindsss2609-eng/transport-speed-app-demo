import os
import time
from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit
import numpy as np

app = Flask(__name__)
app.config['SECRET_KEY'] = 'aero-nexus-telematics-2026'

try:
    import gevent
    import geventwebsocket
    chosen_async_mode = 'gevent'
except ImportError:
    chosen_async_mode = None

socketio = SocketIO(
    app, 
    cors_allowed_origins="*", 
    async_mode=chosen_async_mode, 
    websocket_ping_timeout=40, 
    websocket_ping_interval=10
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

def universal_transport_classifier(speed_kmh, accel_mps2):
    """Universal kinematic classifier scaling seamlessly from walking up to supersonic flight."""
    abs_accel = abs(accel_mps2)
    
    if speed_kmh < 1.0:
        return "Stationary"
    elif speed_kmh <= 7.0:
        return "Walking / Jogging"
    elif speed_kmh <= 30.0:
        return "Biking / Eco-Mobility"
    elif speed_kmh <= 140.0:
        return "Train / Metro" if abs_accel < 0.6 else "Car / Bus"
    elif speed_kmh <= 380.0:
        return "High-Speed Rail / Takeoff Run"
    elif speed_kmh <= 1150.0:
        # Standard commercial aviation cruise profiles fall here
        return "Commercial Airliner Jet"
    else:
        # Speeds breaking the transonic/supersonic thresholds
        return "Supersonic Jet Craft (Mach " + str(round(speed_kmh / 1225.0, 2)) + ")"

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
    if session_id not in device_registry or not device_registry[session_id].get('active', False):
        return
    
    try:
        lat = float(payload.get('lat'))
        lng = float(payload.get('lng'))
    except (TypeError, ValueError):
        return 

    # High-resolution multi-source time parsing 
    raw_ts = payload.get('timestamp')
    ts = float(raw_ts) / 1000.0 if raw_ts else time.time()
    current_time = time.time()
    
    state = device_registry[session_id]
    calculated_speed_kmh = 0.0
    incremental_dist = 0.0
    
    if state['last_lat'] is not None and state['last_lng'] is not None and state['last_ts'] is not None:
        incremental_dist = haversine_distance(state['last_lat'], state['last_lng'], lat, lng)
        time_delta_seconds = ts - state['last_ts']
        
        # Safe execution floor for high-frequency GPS processing strings
        if time_delta_seconds > 0.0001:
            time_delta_hours = time_delta_seconds / 3600.0
            calculated_speed_kmh = incremental_dist / time_delta_hours

            # Prioritize client calculation overrides or hardware pings if available
            client_speed = payload.get('speed')
            if client_speed is not None:
                try:
                    alt_speed_kmh = float(client_speed) * 3.6
                    # If local client calculation indicates structural flight speeds, trust the client vector override
                    if alt_speed_kmh > 150.0 or calculated_speed_kmh < 0.1:
                        calculated_speed_kmh = alt_speed_kmh
                except (TypeError, ValueError):
                    pass

    # Complete removal of the legacy rigid jitter floor. Static lock only applies if standing still.
    if incremental_dist < 0.0001 and calculated_speed_kmh < 0.2:
        calculated_speed_kmh = 0.0
        incremental_dist = 0.0

    # Smooth processing loop for acceleration profiles
    acceleration_mps2 = 0.0
    g_force = 0.0
    
    if state['last_ts'] is not None:
        time_delta = ts - state['last_ts']
        if time_delta > 0.0001:
            dv = (calculated_speed_kmh - state['last_speed']) / 3.6
            acceleration_mps2 = dv / time_delta
            g_force = acceleration_mps2 / 9.80665

    # Append to cumulative tracking odometer 
    if calculated_speed_kmh > 0.2:
        state['total_distance_km'] += incremental_dist

    mode_prediction = universal_transport_classifier(calculated_speed_kmh, acceleration_mps2)

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