import pandas as pd
import numpy as np
import json
import math
import requests
import re
import os
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score

# ──────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────
MAX_DAILY_TRAVEL_KM = 200
ISLAND_STATES = {"Andaman & Nicobar", "Lakshadweep"}
HUB_DISTANCE_THRESHOLD = {
    'railway': 50,   # km
    'airport': 100   # km
}

# ──────────────────────────────────────────────────────────
# 1. LOAD DATASET
# ──────────────────────────────────────────────────────────
filename = "india_tourism_dataset.json"

with open(filename, "r", encoding="utf-8") as f:
    data = json.load(f)

df = pd.DataFrame(data)

df['latitude'] = df['coordinates'].apply(lambda x: x.get('latitude') if isinstance(x, dict) else None)
df['longitude'] = df['coordinates'].apply(lambda x: x.get('longitude') if isinstance(x, dict) else None)
df.dropna(subset=['latitude', 'longitude', 'destination_name', 'state'], inplace=True)

# ----- DEDUPLICATION -----
def clean_name(name):
    if not isinstance(name, str):
        return ""
    name = re.sub(r'\([^)]*\)', '', name)
    name = re.sub(r'[^\w\s]', ' ', name)
    name = re.sub(r'\s+', ' ', name).strip().lower()
    return name

df['name_clean'] = df['destination_name'].apply(clean_name)
df = df.loc[df.groupby('name_clean')['popularity_score'].idxmax()].reset_index(drop=True)
df.drop('name_clean', axis=1, inplace=True)

print(f"After deduplication: {len(df)} unique places")

df['popularity_score'] = df['popularity_score'].fillna(df['popularity_score'].median())

if 'typical_duration' not in df.columns:
    df['typical_duration'] = 2
else:
    df['typical_duration'] = df['typical_duration'].fillna(2)

# Create a mapping from place name to state (for island check)
place_to_state = df.set_index('destination_name')['state'].to_dict()

# ----- FEATURE UTILS -----
def parse_budget(row, category):
    try:
        rng = row.get(category, {}).get('total_daily_range', [0, 0])
        return np.mean(rng) if isinstance(rng, list) and len(rng) == 2 else 0
    except:
        return 0

df['daily_cost_budget'] = df.apply(lambda row: parse_budget(row, 'budget_category'), axis=1)
df['daily_cost_mid'] = df.apply(lambda row: parse_budget(row, 'mid_range_category'), axis=1)
df['daily_cost_luxury'] = df.apply(lambda row: parse_budget(row, 'luxury_category'), axis=1)

accessibility_map = {"Easy": 3, "Moderate": 2, "Difficult": 1, "Hard": 1}
df['accessibility_score'] = df['accessibility'].map(lambda x: accessibility_map.get(x, 2) if isinstance(x, str) else 2)

df['safety_rating'] = pd.to_numeric(df['safety_rating'], errors='coerce').fillna(5)
df['altitude_m'] = pd.to_numeric(df['altitude_m'], errors='coerce').fillna(0)
df['minimum_days'] = pd.to_numeric(df['minimum_days'], errors='coerce').fillna(1)
df['ideal_days'] = pd.to_numeric(df['ideal_days'], errors='coerce').fillna(2)
df['maximum_days'] = pd.to_numeric(df['maximum_days'], errors='coerce').fillna(3)

# ──────────────────────────────────────────────────────────
# 2. GREEN SCORE
# ──────────────────────────────────────────────────────────
green_map = {
    'nature': 9, 'wildlife': 9, 'trekking': 8, 'beach': 7,
    'heritage': 5, 'historical': 5, 'cultural': 5,
    'adventure': 4, 'city': 3, 'shopping': 2
}

def compute_green_score(trip_types):
    if not isinstance(trip_types, list):
        return 5.0
    scores = [green_map.get(t.lower(), 5) for t in trip_types if isinstance(t, str)]
    return np.mean(scores) if scores else 5.0

df['green_score'] = df['trip_types'].apply(compute_green_score)
df['base_score'] = 0.7 * df['popularity_score'] + 0.3 * df['green_score']

# ──────────────────────────────────────────────────────────
# 3. HUB EXTRACTION
# ──────────────────────────────────────────────────────────
hub_info = {}

def extract_hub(row, prefix):
    field = row.get(prefix)
    if isinstance(field, dict):
        name = field.get('name')
        dist = field.get('distance_km')
        if name and dist is not None:
            try:
                dist = float(dist)
                return (name, dist)
            except:
                pass
    name_col = f"{prefix}.name"
    dist_col = f"{prefix}.distance_km"
    if name_col in row and dist_col in row:
        name = row[name_col]
        dist = row[dist_col]
        if pd.notna(name) and pd.notna(dist):
            try:
                dist = float(dist)
                return (name, dist)
            except:
                pass
    return None

for idx, row in df.iterrows():
    dest = row['destination_name']
    airport = extract_hub(row, 'nearest_airport')
    railway = extract_hub(row, 'nearest_railway_station')
    hub_info[dest] = {'airport': airport, 'railway': railway}

print(f"Hub info built for {len(hub_info)} destinations.")
count_with_hub = sum(1 for v in hub_info.values() if v['airport'] or v['railway'])
print(f"Destinations with hub data: {count_with_hub}")

def recommend_hubs(place_name):
    if place_name not in hub_info:
        return "No transport hub data available."
    info = hub_info[place_name]
    parts = []
    if info['airport']:
        name, dist = info['airport']
        parts.append(f"Airport: {name} ({dist} km)")
    if info['railway']:
        name, dist = info['railway']
        parts.append(f"Railway: {name} ({dist} km)")
    if not parts:
        return "No transport hub data available."
    return "; ".join(parts)

# ──────────────────────────────────────────────────────────
# 4. HELPER FUNCTIONS
# ──────────────────────────────────────────────────────────
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def calculate_carbon(dist_km, mode='car'):
    factors = {'car': 0.12, 'bus': 0.08, 'train': 0.06, 'flight': 0.25, 'ferry': 0.05}
    return round(dist_km * factors.get(mode, 0.12), 1)

def suggest_transport_between(place1, place2, dist_km):
    """
    Suggest transport mode based on hub availability at both places.
    place1, place2 are destination names (strings). If a place is a custom start (not in hub_info), fallback to distance.
    Returns (mode, price_estimate)
    """
    # Check if EITHER place is in island states -> ferry / flight
    if place1 in place_to_state or place2 in place_to_state:
        state1 = place_to_state.get(place1)
        state2 = place_to_state.get(place2)
        if state1 in ISLAND_STATES or state2 in ISLAND_STATES:
            if dist_km < 150 or (state1 in ISLAND_STATES and state2 in ISLAND_STATES):
                return 'ferry', int(dist_km * 3)
            else:
                return 'flight', int(dist_km * 5)

    # Get hub info for both places (if available)
    info1 = hub_info.get(place1, {})
    info2 = hub_info.get(place2, {})

    # Check railway availability (both have railway within threshold)
    rail1 = info1.get('railway') if info1 else None
    rail2 = info2.get('railway') if info2 else None
    if rail1 and rail2:
        if rail1[1] <= HUB_DISTANCE_THRESHOLD['railway'] and rail2[1] <= HUB_DISTANCE_THRESHOLD['railway']:
            return 'train', int(dist_km * 1.5)

    # Check airport availability (both have airport within threshold)
    air1 = info1.get('airport') if info1 else None
    air2 = info2.get('airport') if info2 else None
    if air1 and air2:
        if air1[1] <= HUB_DISTANCE_THRESHOLD['airport'] and air2[1] <= HUB_DISTANCE_THRESHOLD['airport']:
            return 'flight', int(dist_km * 5)

    # Fallback to distance-based
    if dist_km < 200:
        return 'bus', int(dist_km * 2)
    elif dist_km < 500:
        return 'train', int(dist_km * 1.5)
    else:
        return 'flight', int(dist_km * 5)

# ... (other helpers: get_driving_distance, adjust_days, clustering utilities, TSP solvers, etc. - all unchanged from previous version)
# For brevity, I'm including the full code with all functions; assume they are present.
# (In practice, the user will replace their existing file with this combined version.)

# ──────────────────────────────────────────────────────────
# 5. DISTANCE CACHE AND GEOAPIFY
# ──────────────────────────────────────────────────────────
_DISTANCE_CACHE_FILE = "distance_cache.json"

def _load_distance_cache():
    if os.path.exists(_DISTANCE_CACHE_FILE):
        try:
            with open(_DISTANCE_CACHE_FILE, "r") as f:
                data = json.load(f)
                return {tuple(map(float, k.split(','))): tuple(v) for k, v in data.items()}
        except Exception:
            return {}
    return {}

def _save_distance_cache(cache):
    try:
        data = {f"{k[0]},{k[1]},{k[2]},{k[3]}": v for k, v in cache.items()}
        with open(_DISTANCE_CACHE_FILE, "w") as f:
            json.dump(data, f)
    except Exception:
        pass

_distance_cache = _load_distance_cache()

def get_driving_distance(origin_lat, origin_lon, dest_lat, dest_lon, api_key=None):
    key = (origin_lat, origin_lon, dest_lat, dest_lon)
    if key in _distance_cache:
        return _distance_cache[key]

    if api_key:
        url = "https://api.geoapify.com/v1/routing"
        params = {
            'waypoints': f"{origin_lat},{origin_lon}|{dest_lat},{dest_lon}",
            'mode': 'drive',
            'apiKey': api_key
        }
        try:
            response = requests.get(url, params=params, timeout=5)
            data = response.json()
            
            if response.status_code != 200:
                print(f"Geoapify HTTP {response.status_code}: {data.get('error', 'Unknown error')}")
            elif 'features' in data and len(data['features']) > 0:
                props = data['features'][0]['properties']
                dist_m = props.get('distance')
                time_s = props.get('time')
                if dist_m is not None and time_s is not None:
                    _distance_cache[key] = (dist_m / 1000, time_s / 3600)
                    _save_distance_cache(_distance_cache)
                    return _distance_cache[key]
                else:
                    print("Geoapify: missing distance/time in properties")
            else:
                print(f"Geoapify: no features found – check API key/permissions")
        except Exception as e:
            print(f"Geoapify error: {e}")

    dist_km = haversine(origin_lat, origin_lon, dest_lat, dest_lon)
    duration_h = dist_km / 50
    _distance_cache[key] = (dist_km, duration_h)
    _save_distance_cache(_distance_cache)
    return dist_km, duration_h

# ──────────────────────────────────────────────────────────
# 6. DAY ADJUSTMENT
# ──────────────────────────────────────────────────────────
def adjust_days(num_days, total_places, max_places_per_day, is_india_full=False):
    required = math.ceil(total_places / max_places_per_day)
    if is_india_full and required < 365:
        required = 365
        msg = f"Info: For a complete India tour, days increased to 365 (but cannot exceed number of places)."
    elif num_days < required:
        msg = f"Info: Your requested {num_days} days is too few to cover all {total_places} places. Extending to {required} days."
    elif num_days > required:
        msg = f"Info: Your requested {num_days} days is more than needed to cover all places. Truncating to {required} days (no extra rest days)."
    else:
        msg = None
    return required, msg

# ──────────────────────────────────────────────────────────
# 7. CLUSTERING UTILITIES
# ──────────────────────────────────────────────────────────
def reassign_empty_clusters(labels, n_clusters):
    unique, counts = np.unique(labels, return_counts=True)
    cluster_counts = dict(zip(unique, counts))
    empty_clusters = [c for c in range(n_clusters) if c not in cluster_counts]
    if not empty_clusters:
        return labels
    largest = max(cluster_counts.items(), key=lambda x: x[1])[0]
    for empty in empty_clusters:
        idx_to_move = np.where(labels == largest)[0][0]
        labels[idx_to_move] = empty
    return labels

def haversine_cluster(cluster_coords, n_clusters):
    if len(cluster_coords) <= 1 or n_clusters == 1:
        return np.zeros(len(cluster_coords), dtype=int)
        
    n = len(cluster_coords)
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            d = haversine(cluster_coords[i][0], cluster_coords[i][1], cluster_coords[j][0], cluster_coords[j][1])
            dist_matrix[i][j] = dist_matrix[j][i] = d

    agg = AgglomerativeClustering(n_clusters=n_clusters, metric='precomputed', linkage='average')
    return agg.fit_predict(dist_matrix)

def cluster_destinations(coords, n_clusters):
    labels = haversine_cluster(coords, n_clusters)
    return labels, None

def split_large_clusters(day_to_indices, coords, max_places_per_day):
    new_day_to_indices = {}
    new_day = 0
    for cluster_id, indices in day_to_indices.items():
        size = len(indices)
        if size <= max_places_per_day:
            new_day_to_indices[new_day] = indices
            new_day += 1
        else:
            n_sub = math.ceil(size / max_places_per_day)
            sub_coords = coords[indices]
            sub_labels = haversine_cluster(sub_coords, n_sub)
            for sub in range(n_sub):
                sub_indices = [indices[i] for i, lbl in enumerate(sub_labels) if lbl == sub]
                new_day_to_indices[new_day] = sub_indices
                new_day += 1
    return new_day_to_indices

def estimate_route_distance(indices, coords):
    if len(indices) <= 1:
        return 0.0
    pts = coords[indices]
    n = len(pts)
    visited = [False] * n
    path = [0]
    visited[0] = True
    current = 0
    total = 0.0
    for _ in range(n-1):
        best_dist = float('inf')
        best_j = -1
        for j in range(n):
            if not visited[j]:
                d = haversine(pts[current][0], pts[current][1], pts[j][0], pts[j][1])
                if d < best_dist:
                    best_dist = d
                    best_j = j
        total += best_dist
        visited[best_j] = True
        current = best_j
    return total

def split_by_distance(day_to_indices, coords, max_km):
    new_day_counter = len(day_to_indices)
    changed = True
    while changed:
        changed = False
        current_days = list(day_to_indices.keys())
        for day in current_days:
            indices = day_to_indices[day]
            if len(indices) <= 1:
                continue
            dist = estimate_route_distance(indices, coords)
            if dist > max_km:
                print(f"  Splitting day {day} (internal est. {dist:.1f} km > {max_km} km)")
                sub_coords = coords[indices]
                sub_labels = haversine_cluster(sub_coords, 2)
                cluster0 = [indices[i] for i, lbl in enumerate(sub_labels) if lbl == 0]
                cluster1 = [indices[i] for i, lbl in enumerate(sub_labels) if lbl == 1]
                del day_to_indices[day]
                day_to_indices[new_day_counter] = cluster0
                new_day_counter += 1
                day_to_indices[new_day_counter] = cluster1
                new_day_counter += 1
                changed = True
                break
    return day_to_indices

# ──────────────────────────────────────────────────────────
# 8. TSP SOLVERS (for points and for days)
# ──────────────────────────────────────────────────────────
def nearest_neighbor_tsp(points, start_idx=0):
    n = len(points)
    if n <= 1:
        return list(range(n)), 0.0
    visited = [False] * n
    path = [start_idx]
    visited[start_idx] = True
    current = start_idx
    total = 0.0
    for _ in range(n-1):
        best_dist = float('inf')
        best_j = -1
        for j in range(n):
            if not visited[j]:
                d = haversine(points[current][0], points[current][1], points[j][0], points[j][1])
                if d < best_dist:
                    best_dist = d
                    best_j = j
        total += best_dist
        path.append(best_j)
        visited[best_j] = True
        current = best_j
    return path, total

def two_opt_tsp(points, initial_path, max_iter=100):
    n = len(points)
    best = initial_path[:]
    improved = True
    for _ in range(max_iter):
        improved = False
        for i in range(1, n-2):
            for j in range(i+1, n):
                if j - i == 1:
                    continue
                new_path = best[:i] + best[i:j][::-1] + best[j:]
                new_dist = 0.0
                for k in range(n-1):
                    a, b = new_path[k], new_path[k+1]
                    new_dist += haversine(points[a][0], points[a][1], points[b][0], points[b][1])
                old_dist = 0.0
                for k in range(n-1):
                    a, b = best[k], best[k+1]
                    old_dist += haversine(points[a][0], points[a][1], points[b][0], points[b][1])
                if new_dist < old_dist:
                    best = new_path
                    improved = True
        if not improved:
            break
    final_dist = 0.0
    for k in range(n-1):
        a, b = best[k], best[k+1]
        final_dist += haversine(points[a][0], points[a][1], points[b][0], points[b][1])
    return best, final_dist

# ──────────────────────────────────────────────────────────
# 9. BUILD A SINGLE DAY (with updated transport suggestion)
# ──────────────────────────────────────────────────────────
def build_day_from_indices(day_indices, places_df, prev_lat, prev_lon, prev_name, api_key):
    day_places_df = places_df.iloc[day_indices].sort_values('final_score', ascending=False)
    day_places = day_places_df['destination_name'].values
    day_coords = day_places_df[['latitude', 'longitude']].values
    day_durations = day_places_df['typical_duration'].values

    # Prepend previous day's end
    all_coords = np.vstack([(prev_lat, prev_lon), day_coords])
    all_names = np.insert(day_places, 0, prev_name)
    all_durations = np.insert(day_durations, 0, 0)

    n = len(all_coords)
    if n < 2:
        return None, prev_lat, prev_lon, prev_name, 0, 0, 0, 0

    # Remove duplicate if start point equals first destination
    if n > 1 and all_names[0] == all_names[1]:
        all_coords = all_coords[1:]
        all_names = all_names[1:]
        all_durations = all_durations[1:]
        n -= 1
        if n < 2:
            return None, prev_lat, prev_lon, prev_name, 0, 0, 0, 0

    dist_matrix = np.zeros((n, n))
    time_matrix = np.zeros((n, n))
    # Force local haversine distance for the NxN matrix to save time (O(n^2) API calls)
    for i in range(n):
        for j in range(i+1, n):
            d_km, t_h = get_driving_distance(all_coords[i][0], all_coords[i][1],
                                              all_coords[j][0], all_coords[j][1], api_key=None)
            dist_matrix[i][j] = dist_matrix[j][i] = d_km
            time_matrix[i][j] = time_matrix[j][i] = t_h

    path = nearest_neighbor(dist_matrix, start=0)
    path = two_opt(path, dist_matrix)

    # Now that path is determined, do O(n) API calls for the actual chosen segments
    day_dist = 0
    day_travel_time = 0
    max_leg_dist = 0
    max_leg_idx = -1
    carbon = 0
    
    for k in range(len(path)-1):
        idx_a = path[k]
        idx_b = path[k+1]
        lat_a, lon_a = all_coords[idx_a]
        lat_b, lon_b = all_coords[idx_b]
        
        # Call Geoapify for exactly this solved leg
        true_d_km, true_t_h = get_driving_distance(lat_a, lon_a, lat_b, lon_b, api_key)
        
        day_dist += true_d_km
        day_travel_time += true_t_h
        
        if true_d_km > max_leg_dist:
            max_leg_dist = true_d_km
            max_leg_idx = k
            
        # Temporarily use default mode factor for carbon calculation sum
        # We will recalculate the full carbon later once mode is definitive
        pass

    day_visit_time = sum(all_durations[i] for i in path if i != 0)
    day_total_time = day_travel_time + day_visit_time

    # Determine the mode for this day based on the longest leg
    # Get the two place names for the longest leg (if both are destinations, not start)
    if max_leg_idx != -1:
        i = path[max_leg_idx]
        j = path[max_leg_idx+1]
        if i == 0 or j == 0:
            # Leg involves the start point – use distance-based suggestion
            mode, price = suggest_transport_between("", "", max_leg_dist)  # fallback
        else:
            name_i = all_names[i]
            name_j = all_names[j]
            mode, price = suggest_transport_between(name_i, name_j, max_leg_dist)
    else:
        # No legs? shouldn't happen
        mode, price = 'car', int(max_leg_dist * 2)

    # Recalculate true carbon using true distances and the final chosen mode
    carbon = 0
    for k in range(len(path)-1):
        idx_a = path[k]
        idx_b = path[k+1]
        lat_a, lon_a = all_coords[idx_a]
        lat_b, lon_b = all_coords[idx_b]
        # Re-fetch from cache (O(1))
        true_d_km, _ = get_driving_distance(lat_a, lon_a, lat_b, lon_b, api_key)
        carbon += calculate_carbon(true_d_km, mode)

    route_names = [all_names[i] for i in path]
    route = " → ".join(route_names)

    end_name = route_names[-1]
    end_lat, end_lon = all_coords[path[-1]]

    original_indices = [day_indices[i-1] for i in path if i != 0]

    day_info = {
        "route": route,
        "distance": round(day_dist, 1),
        "carbon": carbon,
        "mode": mode,
        "price": price,
        "total_time_h": round(day_total_time, 1),
        "end_lat": end_lat,
        "end_lon": end_lon,
        "end_name": end_name,
        "origin_hub_recommendation": recommend_hubs(prev_name) if prev_name != "Custom start – no hub data" else "Custom start – no hub data",
        "destination_hub_recommendation": recommend_hubs(end_name),
        "indices": original_indices
    }
    return day_info, end_lat, end_lon, end_name, day_dist, carbon, price, len(day_indices)

def nearest_neighbor(dist_matrix, start=0):
    n = len(dist_matrix)
    visited = [False] * n
    path = [start]
    visited[start] = True
    current = start
    for _ in range(n-1):
        next_node = min([(j, dist_matrix[current][j]) for j in range(n) if not visited[j]], key=lambda x: x[1])[0]
        path.append(next_node)
        visited[next_node] = True
        current = next_node
    return path

def calculate_path_distance(path, dist_matrix):
    return sum(dist_matrix[path[i]][path[i+1]] for i in range(len(path)-1))

def two_opt(path, dist_matrix, max_iterations=100):
    best = path[:]
    improved = True
    n = len(path)
    for _ in range(max_iterations):
        improved = False
        for i in range(1, n-2):
            for j in range(i+1, n):
                if j - i == 1:
                    continue
                new_path = best[:i] + best[i:j][::-1] + best[j:]
                if calculate_path_distance(new_path, dist_matrix) < calculate_path_distance(best, dist_matrix):
                    best = new_path
                    improved = True
        if not improved:
            break
    return best

# ──────────────────────────────────────────────────────────
# 10. GLOBAL DAY ORDERING
# ──────────────────────────────────────────────────────────
def order_days_global(day_to_indices, places_df, start_lat, start_lon):
    day_keys = list(day_to_indices.keys())
    if len(day_keys) <= 1:
        return day_keys

    reps = []
    for day in day_keys:
        indices = day_to_indices[day]
        if not indices:
            continue
        day_df = places_df.iloc[indices].sort_values('final_score', ascending=False)
        best_idx = day_df.index[0]
        lat = places_df.loc[best_idx, 'latitude']
        lon = places_df.loc[best_idx, 'longitude']
        reps.append((lat, lon))

    all_points = [(start_lat, start_lon)] + reps
    n = len(all_points)
    if n <= 2:
        return day_keys

    dist_mat = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            d = haversine(all_points[i][0], all_points[i][1], all_points[j][0], all_points[j][1])
            dist_mat[i][j] = dist_mat[j][i] = d

    path = [0]
    visited = [False] * n
    visited[0] = True
    current = 0
    for _ in range(n-1):
        next_node = min([(j, dist_mat[current][j]) for j in range(n) if not visited[j]], key=lambda x: x[1])[0]
        path.append(next_node)
        visited[next_node] = True
        current = next_node

    best_path = path[:]
    improved = True
    for _ in range(100):
        improved = False
        for i in range(1, n-2):
            for j in range(i+1, n):
                if j - i == 1:
                    continue
                new_path = best_path[:i] + best_path[i:j][::-1] + best_path[j:]
                new_dist = 0.0
                for k in range(n-1):
                    a, b = new_path[k], new_path[k+1]
                    new_dist += dist_mat[a][b]
                old_dist = 0.0
                for k in range(n-1):
                    a, b = best_path[k], best_path[k+1]
                    old_dist += dist_mat[a][b]
                if new_dist < old_dist:
                    best_path = new_path
                    improved = True
        if not improved:
            break

    ordered_days = [day_keys[best_path[i]-1] for i in range(1, n) if best_path[i] != 0]
    return ordered_days

# ──────────────────────────────────────────────────────────
# 11. MAIN FUNCTION
# ──────────────────────────────────────────────────────────
def generate_itinerary(country="India", num_days=7, max_places_per_day=5,
                       start_location=None, state=None, api_key=None,
                       travel_day_only=False,
                       # Additional Features
                       user_budget=None, # "budget", "mid", "luxury"
                       travel_group=None, # e.g. "Families", "Solo"
                       accessibility_preference=None, # min score, e.g. 2 for Moderate
                       min_safety_rating=None, # e.g. 7
                       max_altitude=None, # e.g. 2000
                       preferred_season=None # e.g. "Winter"
                       ):
    places_df = df.copy()

    # State filter
    if state:
        if isinstance(state, str):
            state = [state]
        pattern = '|'.join([re.escape(s.strip()) for s in state])
        mask = places_df['state'].str.contains(pattern, case=False, na=False, regex=True)
        places_df = places_df[mask]
        if places_df.empty:
            return {"error": "No places found in selected state(s)"}
        selected_region = ', '.join(state)
    else:
        selected_region = country

    # Filter based on strict requirements
    if min_safety_rating is not None:
        places_df = places_df[places_df['safety_rating'] >= min_safety_rating]
    if max_altitude is not None:
        places_df = places_df[places_df['altitude_m'] <= max_altitude]
    if accessibility_preference is not None:
        places_df = places_df[places_df['accessibility_score'] >= accessibility_preference]

    # Initialize final_score
    places_df['final_score'] = places_df['base_score']

    # Boost score dynamically based on user preferences
    if travel_group:
        places_df['final_score'] += places_df['ideal_for'].apply(
            lambda x: 2.0 if isinstance(x, list) and any(travel_group.lower() in str(item).lower() for item in x) else 0.0
        )
    if preferred_season:
        places_df['final_score'] += places_df['best_seasons'].apply(
            lambda x: 1.0 if isinstance(x, list) and any(preferred_season.lower() in str(item).lower() for item in x) else 0.0
        )

    places_df = places_df.sort_values('final_score', ascending=False).reset_index(drop=True)

    total_places = len(places_df)
    if total_places == 0:
        return {"error": "No places after filtering"}

    is_india_full = (state is None and country.lower() == "india")
    effective_places = total_places + (1 if travel_day_only else 0)
    required_days, adjust_msg = adjust_days(num_days, effective_places, max_places_per_day, is_india_full)
    sightseeing_days = required_days - (1 if travel_day_only else 0)

    if travel_day_only and sightseeing_days <= 0:
        travel_day_only = False
        sightseeing_days = required_days
        adjust_msg = (adjust_msg or "") + " Travel day disabled because it would leave no sightseeing days."

    coords = places_df[['latitude', 'longitude']].values
    names = places_df['destination_name'].values
    visit_durations = places_df['typical_duration'].values

    if start_location is None:
        start_lat, start_lon, start_name = 28.6139, 77.2090, "New Delhi (Start)"
        start_is_custom = True
    else:
        start_lat = start_location.get('lat', 28.6139)
        start_lon = start_location.get('lon', 77.2090)
        start_name = start_location.get('name', "Custom Start")
        is_custom_explicit = start_location.get('is_custom', False)
        
        matched = False
        if not is_custom_explicit:
            for i, name in enumerate(names):
                if start_name.lower() in name.lower() or name.lower() in start_name.lower():
                    start_name = name
                    start_lat, start_lon = coords[i]
                    matched = True
                    break
        
        start_is_custom = is_custom_explicit or not matched
        if start_is_custom and "(Start)" not in start_name:
            start_name = f"{start_name} (Start)"

    n_clusters = min(sightseeing_days, total_places)
    labels, centers = cluster_destinations(coords, n_clusters)
    labels = reassign_empty_clusters(labels, n_clusters)

    day_to_indices = {i: [] for i in range(n_clusters)}
    for idx, lab in enumerate(labels):
        day_to_indices[lab].append(idx)

    day_to_indices = split_large_clusters(day_to_indices, coords, max_places_per_day)
    day_to_indices = split_by_distance(day_to_indices, coords, MAX_DAILY_TRAVEL_KM)

    # Build itinerary incrementally with global TSP ordering
    itinerary = []
    total_carbon = 0.0
    total_distance = 0.0
    total_cost = 0

    curr_lat, curr_lon = start_lat, start_lon
    curr_name = start_name
    curr_is_custom = start_is_custom
    day_counter = 1

    if travel_day_only:
        reps = []
        for day, indices in day_to_indices.items():
            if indices:
                day_df = places_df.iloc[indices].sort_values('final_score', ascending=False)
                best_idx = day_df.index[0]
                lat = places_df.loc[best_idx, 'latitude']
                lon = places_df.loc[best_idx, 'longitude']
                reps.append((day, (lat, lon)))
        if reps:
            reps.sort(key=lambda x: haversine(curr_lat, curr_lon, x[1][0], x[1][1]))
            first_day = reps[0][0]
            first_indices = day_to_indices[first_day]
            if first_indices:
                first_place_idx = first_indices[0]
                first_lat = places_df.loc[first_place_idx, 'latitude']
                first_lon = places_df.loc[first_place_idx, 'longitude']
                first_name = places_df.loc[first_place_idx, 'destination_name']

                dist_km, travel_h = get_driving_distance(curr_lat, curr_lon, first_lat, first_lon, api_key)
                mode, price = suggest_transport_between(curr_name, first_name, dist_km)
                carbon = calculate_carbon(dist_km, mode)

                origin_recommend = "Custom start – no hub data" if curr_is_custom else recommend_hubs(curr_name)
                dest_recommend = recommend_hubs(first_name)

                travel_day_entry = {
                    "day": day_counter,
                    "type": "travel",
                    "route": f"{curr_name} → {first_name}",
                    "distance": round(dist_km, 1),
                    "carbon": carbon,
                    "mode": mode,
                    "price": price,
                    "total_time_h": round(travel_h, 1),
                    "origin_hub_recommendation": origin_recommend,
                    "destination_hub_recommendation": dest_recommend,
                    "warning": "Travel only – no sightseeing",
                    "end_lat": first_lat,
                    "end_lon": first_lon,
                    "end_name": first_name
                }
                itinerary.append(travel_day_entry)
                total_carbon += carbon
                total_distance += dist_km
                total_cost += price
                curr_lat, curr_lon = first_lat, first_lon
                curr_name = first_name
                curr_is_custom = False
                day_counter += 1

    remaining_days = list(day_to_indices.keys())
    max_iter = 100
    iter_count = 0

    while remaining_days and iter_count < max_iter:
        iter_count += 1
        ordered = order_days_global({k: day_to_indices[k] for k in remaining_days}, places_df, curr_lat, curr_lon)
        for next_day in ordered:
            if next_day not in remaining_days:
                continue
            indices = day_to_indices[next_day]
            if not indices:
                remaining_days.remove(next_day)
                continue

            day_result = build_day_from_indices(indices, places_df, curr_lat, curr_lon, curr_name, api_key)
            if day_result[0] is None:
                remaining_days.remove(next_day)
                continue

            day_info, new_lat, new_lon, new_name, day_dist, day_carbon, day_price, num_places = day_result

            # Check place count
            if num_places > max_places_per_day:
                print(f"  Splitting day {next_day} (too many places: {num_places} > {max_places_per_day})")
                sub_coords = coords[indices]
                sub_labels = haversine_cluster(sub_coords, 2)
                cluster0 = [indices[i] for i, lbl in enumerate(sub_labels) if lbl == 0]
                cluster1 = [indices[i] for i, lbl in enumerate(sub_labels) if lbl == 1]
                del day_to_indices[next_day]
                remaining_days.remove(next_day)
                new_id0 = max(day_to_indices.keys()) + 1 if day_to_indices else 0
                new_id1 = new_id0 + 1
                day_to_indices[new_id0] = cluster0
                day_to_indices[new_id1] = cluster1
                remaining_days.extend([new_id0, new_id1])
                break

            # Check distance
            if day_dist > MAX_DAILY_TRAVEL_KM and len(indices) > 1:
                print(f"  Splitting day {next_day} (actual {day_dist:.1f} km > {MAX_DAILY_TRAVEL_KM} km)")
                sub_coords = coords[indices]
                sub_labels = haversine_cluster(sub_coords, 2)
                cluster0 = [indices[i] for i, lbl in enumerate(sub_labels) if lbl == 0]
                cluster1 = [indices[i] for i, lbl in enumerate(sub_labels) if lbl == 1]
                del day_to_indices[next_day]
                remaining_days.remove(next_day)
                new_id0 = max(day_to_indices.keys()) + 1 if day_to_indices else 0
                new_id1 = new_id0 + 1
                day_to_indices[new_id0] = cluster0
                day_to_indices[new_id1] = cluster1
                remaining_days.extend([new_id0, new_id1])
                break
            else:
                if day_dist > MAX_DAILY_TRAVEL_KM and len(indices) == 1:
                    day_info["warning"] = f"Long travel distance ({day_dist:.1f} km) – cannot split a single-place day."

                day_entry = {
                    "day": day_counter,
                    "type": "sightseeing",
                    "route": day_info["route"],
                    "distance": day_info["distance"],
                    "carbon": day_info["carbon"],
                    "mode": day_info["mode"],
                    "price": day_info["price"],
                    "total_time_h": day_info["total_time_h"],
                    "origin_hub_recommendation": day_info["origin_hub_recommendation"],
                    "destination_hub_recommendation": day_info["destination_hub_recommendation"],
                    "warning": day_info.get("warning", "OK"),
                    "indices": day_info["indices"],
                    "end_lat": day_info["end_lat"],
                    "end_lon": day_info["end_lon"],
                    "end_name": day_info["end_name"]
                }
                itinerary.append(day_entry)
                total_carbon += day_carbon
                total_distance += day_dist
                total_cost += day_price
                curr_lat, curr_lon = new_lat, new_lon
                curr_name = new_name
                curr_is_custom = False
                day_counter += 1
                remaining_days.remove(next_day)
        else:
            break

    if iter_count >= max_iter:
        return {"error": "Failed to build itinerary within iteration limit."}

    # Dynamic accommodation logic based on budget preference and dataset averages
    if user_budget == 'budget' and 'daily_cost_budget' in df.columns:
        per_day_accommodation = df['daily_cost_budget'].replace(0, np.nan).median()
    elif user_budget == 'luxury' and 'daily_cost_luxury' in df.columns:
        per_day_accommodation = df['daily_cost_luxury'].replace(0, np.nan).median()
    elif 'daily_cost_mid' in df.columns:
        per_day_accommodation = df['daily_cost_mid'].replace(0, np.nan).median()
    else:
        per_day_accommodation = 2000
    
    if pd.isna(per_day_accommodation) or per_day_accommodation == 0:
        per_day_accommodation = 2000
        
    per_day_accommodation = float(per_day_accommodation) # Ensure it's JSON serializable

    num_itinerary_days = len(itinerary)
    total_accommodation = per_day_accommodation * num_itinerary_days
    total_budget = total_cost + total_accommodation

    budget_breakdown = {
        "transport": total_cost,
        "accommodation_food": total_accommodation,
        "total": total_budget,
        "note": "Accommodation & food estimated at ₹2000 per day"
    }

    return {
        "summary_title": f"Your {num_itinerary_days}-Day Eco-Trip in {selected_region}",
        "days_adjusted_message": adjust_msg,
        "total_carbon": round(total_carbon, 1),
        "total_distance": round(total_distance, 1),
        "budget": budget_breakdown,
        "days": itinerary
    }

# ──────────────────────────────────────────────────────────
# 12. TEST BLOCK
# ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    test_result = generate_itinerary(
        num_days=5,
        state="Rajasthan",
        start_location={"lat": 28.6139, "lon": 77.2090, "name": "Delhi"},
        api_key=None,
        travel_day_only=True
    )
    print(json.dumps(test_result, indent=2))
