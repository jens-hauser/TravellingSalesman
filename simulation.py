import numpy as np
import pandas as pd
import time
import geopandas as gpd
import argparse
import matplotlib.pyplot as plt
from utils import distance, str2bool, preprocess_states, preprocess_capitals
plt.rcParams["figure.figsize"] = (20,10)

# Read and preprocess data for us_states and capitals
states = gpd.read_file('data/cb_2018_us_state_5m.shp')
us_capitals = pd.read_csv("data/us-capitals.csv")
states = preprocess_states(states)
coordinates, us_capitals = preprocess_capitals(us_capitals)

def simulated_annealing(coordinates, route=None, epochs=20000, eta=0.98, T=500, simulate=True):
    """ Approximately solves the travelling salesman problem using simulated annealling. 
        Returns the final route and distance. 

    Parameters:
    coordinates (np.array): Contains coordinates (longitude and latitude) of the cities for which a
    shortest path should be found (shape nx2)
    route (np.array): Optional argument with the starting route
    epochs (int): numebr of epochs to run the procedure
    eta (float) in [0,1]: Decay rate, T is descreased with T**eta every 100 epochs
    T (int): Starting temperature
    simulate (bool): Whether or not to visualize the procedure

    Returns:
    np.array: Final route
    int: Distance of final route

    """
    
    # if no route is given, we start with randomly sampled route
    N = coordinates.shape[0]
    if route is None:
        route = np.random.permutation(N)
    
    distances = []
    current_dist = distance(coordinates, route)
    distances.append(current_dist)

    
    for epoch in range(1, epochs+1):
        for idx in np.random.permutation(N): 
            new_route = route.copy() 
            
            # invert a random part of the route
            sample_idx = np.random.randint(0,N)
            if sample_idx >= idx:
                local_route = route[idx:sample_idx]
                new_route[idx:sample_idx] = local_route[::-1]
                
            else:
                local_route = route[sample_idx:idx]
                new_route[sample_idx:idx] = local_route[::-1]
                         
            # calculate distance with proposal route
            new_dist = distance(coordinates, new_route)
            
            # acceptance probability: always accept if new_dist <= current_dist,
            # if new_dist > current_dist, we still accept with probability exp((current_dist-new_dist)/T
            acc = min(1,np.exp((current_dist-new_dist)/T))
            p = np.random.binomial(1,acc)
            if p == 1:
                route = new_route
                current_dist = new_dist
              
        
        distances.append(distance(coordinates, route))
        
        # descrease Temperature every 100 epochs and simulate if desired
        if epoch % 100  == 0:
            T = T ** 0.98
        if simulate and (epoch % 100 == 0 or (epoch % 10 == 0 and epoch < 100)):
            update(coordinates, route, distances[-1], epoch, T)
            plt.pause(0.5)
            

    return np.asarray(route), np.asarray(distances)


def update(coords, route, dist, epoch, T):
    """ Updates the simulation with the current route and infomation about distance, temperature and epoch.

    Parameters:
    coords (np.array): Contains coordinates with longitude and latitude values (shape nx2)
    route (np.array): Order in which the coordinates in coords should be connected
    dist (float): Current distance of the route
    epoch (int): Current epoch
    T (float): Current temperature in the annealing process

    """
    
    coords = coords[route]
    coords = np.concatenate((coords, coords[0, :].reshape(1,-1)), axis=0)
    ax.set_title(f"Optimal Solution: 17085km, Current Solution: {int(dist)}km, Epoch: {epoch}, Temperature: {np.round(T,1)}", fontsize=16)
    ln.set_data(coords[:,0], coords[:,1])
    fig.canvas.draw_idle()

   

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Arguments for algorithm')
    parser.add_argument('--simulate', type=str2bool, default="true", nargs='?', help='Whether or not to visualize the algorithm.')
    args = parser.parse_args()
    
    
    if args.simulate:
        # Prepare background map of United States with Capitals
        fig, ax = plt.subplots()
        ax.set_aspect('equal')
        us_map = states.boundary.plot(ax=ax, color="grey", linewidth=1)
        us_map = us_capitals.plot(ax=ax, color='blue', markersize=80)
        us_map.axis('off')
        ln, = plt.plot([], [], linewidth = 4, linestyle = "--", color = "green")

        # Run algorithm
        route, dist = simulated_annealing(coordinates, simulate=True)
        plt.show()
    else:
        route, dist = simulated_annealing(coordinates, simulate=False)

