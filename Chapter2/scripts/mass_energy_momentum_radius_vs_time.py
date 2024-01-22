import numpy as np
import matplotlib.pylab as plt
from matplotlib.ticker import AutoMinorLocator
import h5py as h5
from constants import *
from plot_style import *
from scipy.interpolate import interp1d
from tqdm import tqdm


def p_final(n_gas):
    return 2.95e5 * np.power(n_gas, -0.16)


def p_final2(n_gas):
    return 4.8e5 * np.power(n_gas, -1.0 / 7.0)

def p_ST(t, n_gas, E_51 = 1.0):
    return 2.21e4 * np.power(E_51,4./5.) * np.power(n_gas, 1./5.) * np.power(t * 1e3, 3./5.)


def do_sedov_analytic(time_bins, dens, nmodel):

    for i in range(time_bins):

        if time_plot_arr[i] > 0.0:

            r_s, P_s, rho_s, v_s, r_shock, _, _, _, _ = sedov(
                time_plot_arr[i] * constants["YEAR_IN_CGS"] * 1e6,
                1e51,
                dens / (constants["H_mass_fraction"] / constants["PROTON_MASS_IN_CGS"]),
                5.0 / 3.0,
                analytic_space_bins,
                3,
            )

            r_analytical[i, nmodel] = r_shock / constants["PARSEC_IN_CGS"] / 1e3

    return

def M_ST(t, n_gas, E_51 = 1.0):

    r_ST = 5.0 * np.power(E_51, 1./5.) * np.power(n_gas, -1./5.) * np.power(t * 1e3, 2./5.) * constants["PARSEC_IN_CGS"]

    # Compute mass in the shell
    Volume = 4.0 * np.pi / 3.0 * r_ST ** 3

    M_shell = (
        n_gas
        / (constants["H_mass_fraction"] / constants["PROTON_MASS_IN_CGS"])
        * Volume
        / constants["SOLAR_MASS_IN_CGS"]
    )

    return M_shell


def plot_energy(ax):

    for idx, key in enumerate(dict_data.keys()):

        ax.plot(
            time_arr[:, idx],
            np.log10(E_th_arr[:, idx]),
            lw=line_properties["linewidth"][idx],
            color=line_properties["colour"][idx],
            zorder=3,
            dashes=(1.5, 1.5),
        )
        ax.plot(
            time_arr[:, idx],
            np.log10(E_kin_arr[:, idx]),
            lw=line_properties["linewidth"][idx],
            color=line_properties["colour"][idx],
            zorder=3,
            dashes=(4, 1.5, 1, 1.5),
        )
        ax.plot(
            time_arr[:, idx],
            np.log10(E_kin_arr[:, idx] + E_th_arr[:, idx]),
            lw=line_properties["linewidth"][idx],
            color=line_properties["colour"][idx],
            zorder=3,
        )

        if idx==1:
            ax.axvline(
                x=t_sf(1e51, densities[idx]),
                color="grey",
                lw=2,
                dashes=(4, 1),
                alpha=0.8,
                zorder=1,
            )

    ax.plot(
         [5,6], [5,6],
         label="$E_{\\rm thermal}$",
         lw=line_properties["linewidth"][idx],
         color="k",
         zorder=3,
         dashes=(1.5, 1.5),
     )
    ax.plot(
         [5,6], [5,6],
         label="$E_{\\rm kinetic}$",
         lw=line_properties["linewidth"][idx],
         color="k",
         zorder=3,
         dashes=(4, 1.5, 1, 1.5),
    )
    ax.plot(
         [5,6], [5,6],
         label="$E_{\\rm total}$",
         lw=line_properties["linewidth"][idx],
         color="k",
         zorder=3,
    )

    ax.legend(loc="upper right", fontsize=LEGEND_SIZE * 1.0, frameon=False)

    ax.set_xlabel("Time since SN [Myr]", fontsize=LABEL_SIZE)
    ax.set_ylabel("log Energy [$10^{51}$ erg]", fontsize=LABEL_SIZE)
    ax.xaxis.set_tick_params(labelsize=LABEL_SIZE)
    ax.yaxis.set_tick_params(labelsize=LABEL_SIZE)

    ax.set_xticks([0.0, 1.0, 2.0, 3.0])
    ax.set_xlim(-0.1, 3.6)
    ax.set_yticks([-2, -1, 0, 1])
    ax.set_ylim(-2.1, 1.1)

    return


def plot_momentum(ax):

    for idx, key in enumerate(dict_data.keys()):

        print(f"Plotting {key}")

        ax.plot(
            time_arr[:, idx],
            np.log10(p_arr[:, idx]),
            lw=line_properties["linewidth"][idx],
            color=line_properties["colour"][idx],
            zorder=3,
        )

        if idx==1:
            ax.axvline(
                x=t_sf(1e51, densities[idx]),
                color="grey",
                lw=2,
                dashes=(4, 1),
                alpha=0.8,
                zorder=1,
            )
            ax.axhline(
                y=np.log10(p_final(densities[idx])),
                color="black",
                lw=3.5,
                dashes=(3.5,1.25,1.25,1.25),
                alpha=0.95,
                zorder=5,
                label="$p_{\\rm final}$ (Kim \& Ostriker 2015)",
            )
            time_analytic = np.linspace(0, 4.1, 1500)
            ax.plot(time_analytic,
                np.log10(p_ST(time_analytic, densities[idx])),
                color="black",
                lw=3,
                dashes=(1.5, 1.5),
                alpha=0.95,
                zorder=3,
                label="ST solution"
           )

    ax.legend(loc="upper right", fontsize=LEGEND_SIZE * 0.97, frameon=False)

    ax.set_xlabel("Time since SN [Myr]", fontsize=LABEL_SIZE)
    ax.set_ylabel("log Momentum [$\\rm M_\\odot$ km s$^{-1}$]", fontsize=LABEL_SIZE)

    ax.xaxis.set_tick_params(labelsize=LABEL_SIZE)
    ax.yaxis.set_tick_params(labelsize=LABEL_SIZE)

    ax.set_xticks([0.0, 1.0, 2.0, 3.0])
    ax.set_xlim(-0.1, 3.6)
    ax.set_yticks([4.5, 5.0, 5.5, 6.0, 6.5])
    ax.set_ylim(4.4, 6.9)

    return


def plot_mass(ax):

    for idx, key in enumerate(dict_data.keys()):
        ax.plot(
            time_arr[:, idx],
            np.log10(M_arr[:, idx]),
            lw=line_properties["linewidth"][idx],
            color=line_properties["colour"][idx],
            zorder=3,
        )

        if idx==1:
            ax.axvline(
                x=t_sf(1e51, densities[idx]),
                color="grey",
                lw=2,
                dashes=(4, 1),
                alpha=0.8,
                zorder=1,
            )

            M_sedov = M_ST(time_arr[:, idx], densities[idx])
            ax.plot(np.concatenate([[-1e-10], time_arr[1:, idx]]),
                    np.concatenate([[1e-10], np.log10(M_sedov[1:])]), color='k', dashes=(1.5, 1.5), lw=3,
                    label="ST solution", zorder=7)

    ax.legend(loc="lower right", fontsize=LEGEND_SIZE * 1.0, frameon=False)

    ax.set_xlabel("Time since SN [Myr]", fontsize=LABEL_SIZE)
    ax.set_ylabel("log Mass [$\\rm M_\\odot$]", fontsize=LABEL_SIZE)
    ax.xaxis.set_tick_params(labelsize=LABEL_SIZE)
    ax.yaxis.set_tick_params(labelsize=LABEL_SIZE)

    ax.set_xticks([0.0, 1.0, 2.0, 3.0])
    ax.set_xlim(-0.1, 3.6)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_ylim(1.9, 5.4)

    return


def plot_radius(ax):

    for idx, key in enumerate(dict_data.keys()):

        ax.plot(
            time_arr_r[:, idx],
            1e3 * r_numerical[:, idx],
            label=key,
            lw=line_properties["linewidth"][idx],
            color=line_properties["colour"][idx],
            zorder=3,
        )

        if idx == 1:

            print("Plot analytic")

            ax.plot(
                time_plot_arr,
                1e3 * r_analytical[:, idx],
                color="k",
                lw=3,
                dashes=(1.5, 1.5),
                label="ST solution",
                zorder=4,
            )

            ax.axvline(
                x=t_sf(1e51, densities[idx]),
                color="grey",
                lw=2,
                dashes=(4, 1),
                alpha=0.8,
                zorder=1,
            )

    ax.legend(loc="lower right", fontsize=LEGEND_SIZE * 1.0, frameon=False)

    ax.set_xlabel("Time since SN [Myr]", fontsize=LABEL_SIZE)
    ax.set_ylabel("Blastwave position [pc]", fontsize=LABEL_SIZE)
    ax.xaxis.set_tick_params(labelsize=LABEL_SIZE)
    ax.yaxis.set_tick_params(labelsize=LABEL_SIZE)

    ax.set_xticks([0.0, 1.0, 2.0, 3.0])
    ax.set_xlim(-0.1, 3.6)
    ax.set_yticks([0.0, 50.0, 100.0, 150.0, 200.0])
    ax.set_ylim(-5, 240)

    return


if __name__ == "__main__":

    LABEL_SIZE = 37

    densities = np.array([0.1, 0.1])

    dict_data = {"high\_res\_n01": "../run01_diffusion_new_one_p_with_centre_higres_sphere",
                 "high\_res\_n01\_nocooling": "../run01_diffusion_new_one_p_with_centre_higres_nocooling_sphere_lowT"}

    N_models = len(dict_data)
    n_snapshot_max_arg = 135
    N_bins = 301
    r_max = 0.3
    print("Max number of snapshots: {:d}".format(n_snapshot_max_arg))

    save_file_name = "./data/energy_vs_time_data"
    print(f" Loading {save_file_name}...")

    try:
        input_file = np.load(f"{save_file_name}.npz")
        E_kin_arr, E_th_arr, time_arr = input_file["arr_0"], input_file["arr_1"], input_file["arr_2"]
        assert np.shape(E_kin_arr) == (n_snapshot_max_arg, N_models)
    except (IOError, AssertionError) as e:
        raise e

    save_file_name = "./data/mass_vs_time_data"
    print(f" Loading {save_file_name}...")

    try:
        input_file = np.load(f"{save_file_name}.npz")
        M_arr, time_arr = input_file["arr_0"], input_file["arr_1"]
        assert np.shape(M_arr) == (n_snapshot_max_arg, N_models)
    except (IOError, AssertionError) as e:
        raise e

    save_file_name = "./data/momentum_vs_time_data"
    print(f" Loading {save_file_name}...")

    try:
        input_file = np.load(f"{save_file_name}.npz")
        p_arr, time_arr = input_file["arr_0"], input_file["arr_1"]
        assert np.shape(p_arr) == (n_snapshot_max_arg, N_models)

    except (IOError, AssertionError) as e:
        raise e

    # Bins for ST solution
    analytic_space_bins = 1000
    analytic_time_bins = 1000
    time_plot_arr = np.linspace(0.0, 4.0, analytic_time_bins)
    r_analytical = np.zeros((analytic_time_bins, N_models))

    magnitudes_analyt = np.linspace(0.0, r_max, analytic_space_bins + 1)
    magnitudes_centres_analyt = 0.5 * (magnitudes_analyt[:-1] + magnitudes_analyt[1:])

    save_file_name = "./data/radius_vs_time_data"
    print(f" Loading {save_file_name}...")

    for counter in range(len(dict_data)):
        do_sedov_analytic(analytic_time_bins, densities[counter], counter)

    try:
        input_file = np.load(f"{save_file_name}.npz")
        r_numerical, h_numerical, time_arr_r = (
            input_file["arr_0"],
            input_file["arr_1"],
            input_file["arr_2"],
        )
        assert np.shape(r_numerical) == (n_snapshot_max_arg-1, N_models)

    except (IOError, AssertionError) as e:
        raise e

    fig, ax = plt.subplots(2, 2)
    fig.subplots_adjust(hspace=0.28, wspace=0.28)
    fig.set_size_inches(17, 17)

    plt.rc("text", usetex=True)
    plt.rc("font", family="serif")
    plt.rcParams["ytick.direction"] = "in"
    plt.rcParams["xtick.direction"] = "in"
    plt.rcParams["axes.linewidth"] = 2

    for i in range(2):
        for j in range(2):
            ax[i,j].tick_params(which="both", width=1.7)
            ax[i,j].tick_params(which="major", length=9)
            ax[i,j].tick_params(which="minor", length=5)
            ax[i,j].xaxis.set_minor_locator(AutoMinorLocator(5))
            ax[i,j].yaxis.set_minor_locator(AutoMinorLocator(5))
            ax[i,j].tick_params(
                axis="both", which="both", pad=8, left=True, right=True, top=True, bottom=True,
            )
    
    print("PLOT 1")
    plot_energy(ax[0,0])
    print("PLOT 2")
    plot_momentum(ax[0,1])
    print("PLOT 3")
    plot_mass(ax[1,0])
    print("PLOT 4")
    plot_radius(ax[1,1])           

    plt.savefig("./images/Fig1.pdf", bbox_inches="tight", pad_inches=0.1)


