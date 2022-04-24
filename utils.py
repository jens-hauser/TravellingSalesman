import numpy as np
import pandas as pd
import geopandas as gpd
import argparse

def preprocess_states(states):
    """ Preprocesses the states data from Geopandas by ignoring some regions and having the same latitude longitude formate as the cities

    Parameters:
    states (geopandas dataframe): Geopandas DataFrame with information about the different states

    Returns:
    geopandas dataframe: Preprocessed version of states
    """
    states = states.to_crs("EPSG:4326")
    states_to_ignore = [
        "Alaska",
        "Hawaii",
        "Puerto Rico",
        "Commonwealth of the Northern Mariana Islands",
        "Guam",
        "United States Virgin Islands",
        "American Samoa"
    ]
    states = states[~states["NAME"].isin(states_to_ignore)]
    return states

def preprocess_capitals(us_capitals):
    """ Preprocesses the us_capitals data from Geopandas by ignoring some regions and extracts the coordinates into a numpy array
    Parameters:
    states (geopandas dataframe): Geopandas DataFrame with information about the different states

    Returns:
    coordinates (np.array): Coordinates of the cities in longitude, latitude format
    us_capitals (geopandas dataframe): Preprocessed version of us_capitals

    """
    us_capitals = us_capitals[~us_capitals.name.isin(["Alaska", "Hawaii"])]
    us_capitals = gpd.GeoDataFrame(
        us_capitals,
        geometry=gpd.points_from_xy(us_capitals.longitude, us_capitals.latitude)
    )
    us_capitals.reset_index(drop=True, inplace=True)
    coordinates = np.concatenate((np.array(us_capitals.longitude).reshape(-1,1),
                                  np.array(us_capitals.latitude).reshape(-1,1)), axis=1)
    
    return coordinates, us_capitals

def str2bool(v):
    """ Helper to convert a string boolean value to a bool
    
    Parameters:
    v (string): Value corresponnding to a bool

    Returns:
    bool: v converted to bool according to logic

    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def distance(coords, route=None):
    """ Calculates the distance to connect all points in coordinates in the order determined by route.
    From the last point we go back to the first. 

    Parameters:
    coords (np.array): Contains coordinates with longitude and latitude values (shape nx2)
    route (np.array): Optional argument with the order in which the coordinates in coords should be connected

    Returns:
    int: Distance to connect all points in circular fashion

    """
    
    # Radius of earth in kilometers
    r = 6371
    
    # get coords in correct order, additionally we need to get back to the starting point
    if route is not None:
        coords = coords[route]
    coords = np.concatenate((coords, coords[0, :].reshape(1,-1)), axis=0)
    
    # convert longitude and latitude to radians
    coords_rad = coords / (180/np.pi)
      
    # Haversine formula
    diff = np.diff(coords_rad, axis=0)
    a = np.sin(diff[:,1] / 2)**2 + np.cos(coords_rad[:-1,1]) * np.cos(coords_rad[1:,1]) * np.sin(diff[:,0] / 2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    # calculate the result
    return np.sum(c * r)