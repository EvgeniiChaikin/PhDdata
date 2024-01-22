import numpy as np

import matplotlib.pylab as plt
from matplotlib.ticker import AutoMinorLocator

import h5py as h5
import sys
import json
from scipy import stats
from matplotlib import cm as cmm
from matplotlib import ticker
from constants import *
from plot_style import *

def read_sim_data(file: str = "simulations.json"):
    try:
        with open(file, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def plot():

    runs = read_sim_data("main.json")

    for script_name, plots in runs.items():

        print(script_name)

        if script_name == sys.argv[0]:

            print("FOUND")

            for plot in plots:

                output = plot["output_file"]
                dict_sim = plot["data"]
                split = plot["split"]
                snapshot = plot["snapshot"]
                
                print(output)

                bins = 100
                edges = np.linspace(0.0, 4.0, bins) # log km/s
                centers = 0.5 * (edges[1:] + edges[:-1])

                fig, ax = plot_style(8, 8)

                for counter, (key, value) in enumerate(dict_sim.items()):
 
                    f = h5.File(value + "/output_{:04d}.hdf5".format(snapshot), "r")
            
                    unit_length_in_cgs = f["/Units"].attrs["Unit length in cgs (U_L)"]
                    unit_mass_in_cgs = f["/Units"].attrs["Unit mass in cgs (U_M)"]
                    unit_time_in_cgs = f["/Units"].attrs["Unit time in cgs (U_t)"]
            
                    vgas = f["/PartType0/Velocities"][:,:] * unit_length_in_cgs / unit_time_in_cgs / 1e5
                    vgas_abs = np.sqrt(np.sum(vgas**2, axis=1))
                    mass = f["/PartType0/Masses"][:] 

                    values, _, _ = stats.binned_statistic(
                            np.log10(vgas_abs), mass, statistic="sum", bins=edges
                    )
                    print(np.max(values))
                    values /= np.sum(values)                    
                    print(np.max(values))
                    ax.plot(centers, np.log10(1e-10 + values), label=key.replace("_","\_"), 
                                                      lw=lw6[counter],
                                                      color=color6[counter],
                                                      dashes=dashes6[counter])
                    time = f["/Header"].attrs["Time"]
                    time_snp = time * unit_time_in_cgs / constants["YEAR_IN_CGS"] / 1e6
           
                ax.set_xticks([-1, 0, 1, 2, 3, 4]) 
                ax.set_ylim(-4, 0.5)
                ax.set_xlim(0.0, 4.0)

                ax.xaxis.set_tick_params(labelsize=30)
                ax.yaxis.set_tick_params(labelsize=30)

                ax.set_xlabel("log Gas velocity $v_{\\rm gas}$ [km s$^{-1}$]", fontsize=28)
                ax.set_ylabel("log Mass fr. of particles per bin", labelpad=0, fontsize=28)
            
                #plt.legend(loc="upper left", fontsize=18, frameon=False)
            
                #fixlogax(ax,"x")
                #fixlogax(ax,"y")
            
                plt.savefig(f"./images/{output}", bbox_inches='tight', pad_inches=0.1)
                plt.close()

if __name__ == "__main__":
    plot()            
