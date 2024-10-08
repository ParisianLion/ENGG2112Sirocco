def geo_to_cartesian(latitude, longitude, altitude=0, radius=6371):
    """
    Convert geographic coordinates (latitude, longitude, altitude) to Cartesian coordinates (x, y, z).
    
    Args:
        latitude (float): Latitude in degrees.
        longitude (float): Longitude in degrees.
        altitude (float): Altitude in kilometers (default is 0 km).
        radius (float): Radius of the Earth in kilometers (default is 6371 km).
    
    Returns:
        tuple: Cartesian coordinates (x, y, z).
    """
    # Convert degrees to radians
    lat_rad = math.radians(latitude)
    lon_rad = math.radians(longitude)

    # Effective radius considering altitude
    effective_radius = radius + altitude

    # Calculate Cartesian coordinates
    x = effective_radius * math.cos(lat_rad) * math.cos(lon_rad)
    y = effective_radius * math.cos(lat_rad) * math.sin(lon_rad)
    z = effective_radius * math.sin(lat_rad)

    return (x, y, z)


# Define action map with separate variables for control actions
ACTIONS = {
    0: {"name": "Increase Elevator Angle", "control": "ELEVATOR_TRIM_UP", "value": 1.0},
    1: {"name": "Decrease Elevator Angle", "control": "ELEVATOR_TRIM_DOWN", "value": -1.0},
    2: {"name": "Increase Aileron Angle", "control": "AILERON_TRIM_RIGHT", "value": 1.0},
    3: {"name": "Decrease Aileron Angle", "control": "AILERON_TRIM_LEFT", "value": -1.0},
    4: {"name": "Increase Rudder Angle", "control": "RUDDER_TRIM_RIGHT", "value": 1.0},
    5: {"name": "Decrease Rudder Angle", "control": "RUDDER_TRIM_LEFT", "value": -1.0},
    6: {"name": "No Action", "control": None, "value": 0.0}
}
