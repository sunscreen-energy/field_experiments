import json

corners = {
    'topleft':  (38.367917, -121.616448),
    'topright': (38.367281, -121.613817),
    'botleft':  (38.363171, -121.616451),
    'botright': (38.363120, -121.613840)
}

sensor_locations_normalized = [
    (22/227, 30/526),   # Bottom-left-ish
    (22/227, 163/526),
    (22/227, 312/526),
    (22/227, 460/526),   # Top-left-ish
    ((22 + 90)/227, 90/526),  # bottom middle-ish
    ((22 + 90)/227, 236/526),
    ((22 + 90)/227, 382/526),   # Top-middle-ish
    ((227 - 22)/227, 30/526),   # Bottom-right-ish
    ((227 - 22)/227, 163/526),
    ((227 - 22)/227, 312/526)    # Top-right-ish
    (-3/4, 1),   # Control 1 (NW)
    (1, 1),   # Control 2 (NE)
]

sensor_ids_in_location = [
    [15, 21, 28, 46, 63, 69, 75, 93],
    [10, 13, 45, 55, 56, 64, 87, 97],
    [18, 38, 49, 59, 71, 76, 84, 85],
    [2, 26, 29, 37, 39, 60, 79, 96],
    [11, 12, 35, 61, 70, 88, 89, 91],
    [3, 14, 25, 30, 36, 53, 86, 90],
    [9, 24, 31, 33, 52, 92, 99, 100],
    [4, 22, 40, 44, 48, 67, 78, 98],
    [19, 23, 32, 54, 57, 80, 81, 94],
    [5, 20, 34, 42, 43, 47, 50, 72],
    [1, 17, 58, 65, 66, 74, 82, 83],
    [6, 7, 8, 16, 27, 62, 77, 95],
]

def calculate_sensor_coords(corner_coords, normalized_sensors):
    """
    Computes sensor coordinates using simple Linear Interpolation (LERP)
    within the field's geographic bounding box.
    
    This method assumes the field is a perfect rectangle aligned with
    latitude and longitude lines.

    Args:
        corner_coords (dict): A dictionary with keys 'topleft', 'topright', 
                              'botleft', 'botright'. Each value is a 
                              tuple of (latitude, longitude).
        normalized_sensors (list): A list of (fr_x, fr_y) tuples, where
                                   fr_x is the fractional width (0.0 to 1.0)
                                   fr_y is the fractional height (0.0 to 1.0)
                                   (0, 0) corresponds to the bottom-left.

    Returns:
        list: A list of (latitude, longitude) tuples for each sensor.
    """
    
    # Extract all latitudes and longitudes to find the bounding box
    lats = [c[0] for c in corner_coords.values()]
    lons = [c[1] for c in corner_coords.values()]
    
    min_lat = min(lats)
    min_lon = min(lons)
    
    total_height_deg = max(lats) - min_lat
    total_width_deg = max(lons) - min_lon
    
    sensor_locations = []
    for fr_x, fr_y in normalized_sensors:

        sensor_lat = min_lat + fr_y * total_height_deg
        sensor_lon = min_lon + fr_x * total_width_deg
        
        sensor_locations.append((sensor_lat, sensor_lon))
        
    return sensor_locations


calculated_locations = calculate_sensor_coords(corners, sensor_locations_normalized)

sensor_coordinates = {}
for i, (lat, lon) in enumerate(calculated_locations):
    for sensor_id in sensor_ids_in_location[i]:
        sensor_coordinates[f"UNIT_{sensor_id}"] = (lat, lon)

with open('sensor_coordinates.json', 'w') as f:
    json.dump(sensor_coordinates, f, indent=4)
