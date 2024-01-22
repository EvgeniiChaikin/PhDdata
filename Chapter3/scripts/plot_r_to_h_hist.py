import numpy as np
import sys
from scipy import stats
from plot_style import *
import json 
from swiftsimio import load

def read_sim_data(file: str = "simulations.json"):
    try:
        with open(file, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def plot():
    
    gamma_quartic = 2.018932
    runs = read_sim_data("main.json")

    for script_name, plots in runs.items():

        if script_name == sys.argv[0]:

            print("FOUND")

            for plot in plots:
           
                output = plot["output_file"]
                dict_sim = plot["data"]
                idx = plot["snapshot"]
                TIME_WINDOW_MYR = plot["timewindow"]
        
                print(output)
         
                fig, ax = plot_style(8, 8)
 
                for counter, (key, value) in enumerate(dict_sim.items()):
          
                    print(counter, key)
        
                    # Load data
                    data = load(value + "/output_{:04d}.hdf5".format(idx))
        
                    h = data.gas.last_sniithermal_feedback_particle_h
                    h_stars = data.stars.last_sniithermal_feedback_particle_h
                    r = data.gas.last_sniithermal_feedback_particle_r
                    r_stars = data.stars.last_sniithermal_feedback_particle_r
        
                    # Snapshot time 
                    snp_time_Myr = data.metadata.time.to("Myr")
                    
                    # Time last last snii feedback event
                    snii_th_feedback_times = data.gas.last_sniithermal_feedback_times
                    snii_th_feedback_times_stars = data.stars.last_sniithermal_feedback_times
        
                    # Create mask
                    mask =       np.logical_and(snii_th_feedback_times       > 0.0 * snii_th_feedback_times.units,
                                          np.abs(snii_th_feedback_times.to("Myr")      -snp_time_Myr) < TIME_WINDOW_MYR)
                    mask_stars = np.logical_and(snii_th_feedback_times_stars > 0.0 * snii_th_feedback_times_stars.units,
                                          np.abs(snii_th_feedback_times_stars.to("Myr")-snp_time_Myr) < TIME_WINDOW_MYR)
        
                    # Gas + stars
                    r_all = np.concatenate([r[mask], r_stars[mask_stars]])
                    h_all = np.concatenate([h[mask], h_stars[mask_stars]])
       
 
                    bins = 41
                    edges = np.linspace(0.0, 1.0, bins) # Myr
                    centers = 0.5 * (edges[1:] + edges[:-1])

                    values, _, _ = stats.binned_statistic(
                        r_all / h_all / gamma_quartic, np.ones_like(r_all), statistic="sum", bins=edges
                    )

                    values /= np.sum(values)
                    values *= (bins-1.0)

                    # Plot
                    ax.plot(np.concatenate([centers, [centers[-1]+1e-5]]), 
                            np.concatenate([values,  [0.0]]),
                                             label=key.replace("_","\_"),
                                             ls=line_properties["ls"][counter],
                                             color=line_properties["colour"][counter],
                                             lw=line_properties["linewidth"][counter])
       
                

                x = np.linspace(0, 1, bins-1)
                values = x**2
                values /= np.sum(values)
                values *= (bins-1.0)

                ax.plot(np.concatenate([centers, [centers[-1]+1e-5]]), 
                        np.concatenate([values,  [0.0]]),
                                       label="$f(x) \\propto x^2$",
                                       color="grey",
                                       lw=3.0,
                                       dashes=(3, 3))

                ax.set_xlim(-0.05, 1.15)
                ax.set_ylim(-0.1, 3.1)
                ax.set_xlabel("$r/h$", fontsize=33)
                ax.set_ylabel("Probability density", fontsize=33)
        
                plt.xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
                plt.yticks([0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
        
                ax.xaxis.set_tick_params(labelsize=33)
                ax.yaxis.set_tick_params(labelsize=33)
        
                ax.legend(fontsize=22, ncol=2, bbox_to_anchor =(-0.25, 1.25), loc="upper left")
        
                ax.text(
                    0.8, 0.97, "${:.1f} < t < {:.1f}$ Gyr".format( (snp_time_Myr.value - TIME_WINDOW_MYR) / 1e3, snp_time_Myr.value / 1e3),
                    ha="right", va="top", transform=ax.transAxes, fontsize=LABEL_SIZE * 0.9
                )
        
                plt.savefig(f"./images/{output}", bbox_inches='tight', pad_inches=0.1)

                plt.close()

plot()
