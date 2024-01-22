"""
Code from Folkert Nobels to load observational SFR data.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

def bin_data_general(array_x, array_y, array_x_bin, x_limit):
    # create a SFR value array
    y_array_bin = np.zeros(len(array_x_bin)-1)
    y_array_bin_std_up = np.zeros(len(array_x_bin)-1)
    y_array_bin_std_down = np.zeros(len(array_x_bin)-1)

    for i in range(0,len(array_x_bin)-1):
        mask = (array_x > array_x_bin[i]) & (array_x < array_x_bin[i+1])
        y_array_bin[i] = np.median(array_y[mask])
        try:
            y_array_bin_std_up[i], y_array_bin_std_down[i] = np.transpose(np.percentile(array_y[mask], [16,84])) 
        except: 
            y_array_bin_std_up[i], y_array_bin_std_down[i] = [0., 0.]

    array_x_bin =  (array_x_bin[1:] + array_x_bin[:-1])/2.
    y_array_bin_std_up = np.abs(y_array_bin_std_up - y_array_bin)
    y_array_bin_std_down = np.abs(y_array_bin_std_down - y_array_bin)
    mask = array_x_bin > x_limit
    
    return array_x_bin[mask], y_array_bin[mask], y_array_bin_std_up[mask], y_array_bin_std_down[mask]
    


class ObservationalData(object):
    """
    Holds observational data.
    """

    def __init__(self, HI_surface_density, HI_surface_density_error, H2_surface_density, H2_surface_density_error, gas_surface_density, gas_surface_density_error, SFR_surface_density, SFR_surface_density_error, depl_time, description):
        """
        Store stuff in me!
        """
        self.HI_surface_density = HI_surface_density
        self.HI_surface_density_error = HI_surface_density_error
        self.H2_surface_density = H2_surface_density
        self.H2_surface_density_error = H2_surface_density_error
        self.gas_surface_density = gas_surface_density
        self.gas_surface_density_error = gas_surface_density_error
        self.SFR_surface_density = SFR_surface_density
        self.SFR_surface_density_error = SFR_surface_density_error
        self.tdepl = depl_time
        self.description = description

    def bin_data_KS(self, surface_density_bin_array, surface_density_limit):
        '''
            Return the binned KS relation for the observation.
        '''
        obs_gas_surface_density = np.log10(self.gas_surface_density)
        obs_SFR_surface_density = np.log10(self.SFR_surface_density)

        return bin_data_general(obs_gas_surface_density, obs_SFR_surface_density, surface_density_bin_array, surface_density_limit)

    def bin_data_KS_molecular(self, surface_density_bin_array, surface_density_limit):
        '''
            Return the binned KS relation for the observation.
        '''
        obs_gas_surface_density = np.log10(self.H2_surface_density)
        obs_SFR_surface_density = np.log10(self.SFR_surface_density)

        return bin_data_general(obs_gas_surface_density, obs_SFR_surface_density, surface_density_bin_array, surface_density_limit)

    def bin_data_KS_atomic(self, surface_density_bin_array, surface_density_limit):
        '''
            Return the binned KS relation for the observation.
        '''
        obs_gas_surface_density = np.log10(self.HI_surface_density)
        obs_SFR_surface_density = np.log10(self.SFR_surface_density)

        return bin_data_general(obs_gas_surface_density, obs_SFR_surface_density, surface_density_bin_array, surface_density_limit)


    def bin_data_gas_depletion(self, surface_density_bin_array, surface_density_limit):
        '''
            Return the binned KS gas depletion relation for the observation.
        '''
        # Get the errays we want to get the binning 
        obs_gas_surface_density = np.log10(self.gas_surface_density)
        obs_SFR_surface_density = np.log10(self.SFR_surface_density)
        obs_gas_consumption = obs_gas_surface_density - obs_SFR_surface_density + 6

        return bin_data_general(obs_gas_surface_density, obs_gas_consumption, surface_density_bin_array, surface_density_limit)

    def bin_data_gas_depletion_molecular(self, surface_density_bin_array, surface_density_limit):
        '''
            Return the binned KS gas depletion relation for the observation.
        '''
        # Get the errays we want to get the binning 
        obs_gas_surface_density = np.log10(self.gas_surface_density)
        obs_SFR_surface_density = np.log10(self.SFR_surface_density)
        obs_gas_consumption = obs_gas_surface_density - obs_SFR_surface_density + 6

        return bin_data_general(obs_gas_surface_density, obs_gas_consumption, surface_density_bin_array, surface_density_limit)

    def bin_data_gas_depletion_atomic(self, surface_density_bin_array, surface_density_limit):
        '''
            Return the binned KS gas depletion relation for the observation.
        '''
        # Get the errays we want to get the binning 
        obs_gas_surface_density = np.log10(self.HI_surface_density)
        obs_SFR_surface_density = np.log10(self.SFR_surface_density)
        obs_gas_consumption = obs_gas_surface_density - obs_SFR_surface_density + 6

        return bin_data_general(obs_gas_surface_density, obs_gas_consumption, surface_density_bin_array, surface_density_limit)


def read_obs_data(path="obs_data"):

    output = []

    # Load the data from Kennicutt 1998 (part 1, Normal spiral galaxies)
    galaxy_name, D, sigma_HI, sigma_H2, sigma_gas, sigma_SFR, tdepl = np.genfromtxt(f"{path}/KS_relation_Kennicutt1998.txt", unpack=True)

    output.append(
        ObservationalData(
            10**sigma_HI, None, 10**sigma_H2, None, 10**sigma_gas, None, 10**sigma_SFR/1.65, None, tdepl, "Kennicutt (1998) [Normal spirals]"
        )
    )

    # Load the data from Kennicutt 1998 (part 2, circumnuclear starbursts)
    D, sigma_H2, sigma_SFR, tdepl = np.genfromtxt(f"{path}/KS_relation_Kennicutt1998b.txt", unpack=True, usecols = (2,3,4,5))
    
    output.append(
        ObservationalData(
            None, None, 10**sigma_H2, None, 10**sigma_H2, None, 10**sigma_SFR/1.65, None, tdepl, "Kennicutt (1998) [Starbursts]"
        )
    )
 
    # Load the data from Freundlich et al. 2013 (high redshift z=1.2 galaxies)
    Mgas, SFR, sigma_gas, sigma_SFR, tdepl = np.genfromtxt(f"{path}/KS_relation_freundlich2013.txt", unpack=True, usecols = (2,3,4,5,6))    

    output.append(
        ObservationalData(
            None, None, None, None, 10**sigma_gas, None, 10**sigma_SFR, None, tdepl, "Freundlich et al. (2013) [z=1.2 galaxies]"
        )
    )    

    # Load the data from Leroy et al. (2008) and Frank et al. (2016)
    sigma_HI, sigma_HI_err, sigma_H2, sigma_H2_err, sigma_SFR, sigma_SFR_err = np.genfromtxt(f"{path}/surfdens_online.txt", unpack=True, usecols = (2,3,4,5,6,7))    

    sigma_gas = sigma_H2 + sigma_HI
    
    output.append(
        ObservationalData(
            sigma_HI, sigma_HI_err, sigma_H2, sigma_H2_err, sigma_gas, None, sigma_SFR*1e-4, sigma_SFR_err*1e-4, tdepl, "Leroy et al. (2008) and Frank et al. (2016)"
        )
    )
    
    sigma_H2, sigma_SFR = np.genfromtxt(f"{path}/KS_relation_genzel2010.txt", unpack=True, usecols=(11,13))
    sigma_gas = sigma_H2

    output.append(
        ObservationalData(
            None, None, 10**sigma_H2, None, 10**sigma_gas, None, 10**sigma_SFR, None, None, "Genzel et al. (2010) [SFG z=1-3]"
        )
    )

    with open(f"{path}/KS_relation_Bigiel2010.txt") as f:
        lines = f.readlines()
    
    size_array = len(lines) - 49

    sigma_HI = -5*np.ones(size_array)
    sigma_HI_err = -5*np.ones(size_array)
    sigma_H2 = -5*np.ones(size_array)
    sigma_H2_err = -5*np.ones(size_array)
    sigma_SFR = -5*np.ones(size_array)
    sigma_SFR_err = -5*np.ones(size_array)

    for i in range(49, len(lines)):
        k = i - 49
        word1 = lines[i][25:29]
        word2 = lines[i][30:34]
        word3 = lines[i][35:39]
        word4 = lines[i][40:44]
        word5 = lines[i][45:50]
        word6 = lines[i][52:56]
        if word1 != "    ":
            sigma_HI[k] = float(word1)
        if word2 != "    ":
            sigma_HI_err[k] = float(word2)
        if word3 != "    ":
            sigma_H2[k] = float(word3)
        if word4 != "    ":
            sigma_H2_err[k] = float(word4)
        if word5 != "     ":
            sigma_SFR[k] = float(word5)
        if word6 != "    ":
            sigma_SFR_err[k] = float(word6)
 
    sigma_gas = 10**sigma_H2 + 10**sigma_HI

    output.append(
        ObservationalData(
            10**sigma_HI/1.36, sigma_HI_err, 10**sigma_H2/1.36, sigma_H2_err, sigma_gas/1.36, None, 10**sigma_SFR, sigma_SFR_err, None, "Bigiel et al. (2008) inner"
        )
    )
    
    with open(f"{path}/Bigiel_et_al_2010_outer_part.txt") as f:
        lines = f.readlines()
    
    size_array = len(lines) - 46

    sigma_HI = -5.2*np.ones(size_array)
    sigma_HI_err = -5.2*np.ones(size_array)
    sigma_SFR = 10**-0.2*np.ones(size_array)
    sigma_SFR_err = 10**-0.2*np.ones(size_array)

    for i in range(49, len(lines)):
        k = i - 49
        word1 = lines[i][20:25]
        word2 = lines[i][26:30]
        word3 = lines[i][31:37]
        word4 = lines[i][38:42]
        if word1 != "     ":
            sigma_HI[k] = float(word1)
        if word2 != "    ":
            sigma_HI_err[k] = float(word2)
        if word3 != "      ":
            sigma_SFR[k] = float(word3)
        if word4 != "    ":
            sigma_SFR_err[k] = float(word4)

    sigma_gas = 10**sigma_HI

    output.append(
        ObservationalData(
            10**sigma_HI/1.36, sigma_HI_err, None, None, sigma_gas/1.36, None, sigma_SFR*1e-5, sigma_SFR_err*1e-5, None, "Bigiel et al. (2010) outer"
        )
    )

    with open(f"{path}/KS_relation_Leroy2008.txt") as f:
        lines = f.readlines()
    
    size_array = len(lines) - 34

    sigma_HI = -5*np.ones(size_array)
    sigma_HI_err = -5*np.ones(size_array)
    sigma_H2 = -5*np.ones(size_array)
    sigma_H2_err = -5*np.ones(size_array)
    sigma_SFR = -5*np.ones(size_array)
    sigma_SFR_err = -5*np.ones(size_array)

    for i in range(34, len(lines)):
        k = i - 34
        word1 = lines[i][18:22]
        word2 = lines[i][24:27]
        word3 = lines[i][28:33]
        word4 = lines[i][36:39]
        word5 = lines[i][54:60]
        word6 = lines[i][64:67]

        if word1 != "    ":
            sigma_HI[k] = float(word1)
        if word2 != "   ":
            sigma_HI_err[k] = float(word2)
        if word3 != "     ":
            sigma_H2[k] = float(word3)
        if word4 != "   ":
            sigma_H2_err[k] = float(word4)
        if word5 != "      ":
            sigma_SFR[k] = float(word5)
        if word6 != "   ":
            sigma_SFR_err[k] = float(word6)
 
    sigma_gas = sigma_H2 + sigma_HI

    output.append(
        ObservationalData(
            sigma_HI, sigma_HI_err, sigma_H2, sigma_H2_err, sigma_gas, None, sigma_SFR*1e-4, sigma_SFR_err*1e-4, None, "Leroy et al. (2008)"
        )
    )    

    sigma_SFR, sigma_SFR_err, sigma_HI, sigma_HI_err, sigma_H2, sigma_H2_err, quality = np.genfromtxt(f"{path}/data_schruba2011.txt", unpack=True, usecols=(3,4,5,6,7,8,10))

    sigma_gas = sigma_HI + sigma_H2

    mask = quality>-1
    
    output.append(
        ObservationalData(
            sigma_HI[mask], sigma_HI_err[mask], sigma_H2[mask], sigma_H2_err[mask], sigma_gas[mask], None, sigma_SFR[mask], sigma_SFR[mask], None, "Schruba et al. (2011)"
        )
    )


    return output

