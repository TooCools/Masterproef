from math import sin, pi, cos
import noise
import numpy as np
from xsrc.simulation.params import *

slope_offset = 19


def bicycle_model(t_dc_array, fcc_array):
    avg_tdc = np.average(t_dc_array[-50:])
    # fcc = 7 * t_dc_array[-1:][0] - 10
    fcc = 7 * avg_tdc - 10
    if fcc < 0:
        fcc = 0
    elif fcc > 120:
        fcc = 120
    fcc = int(5 * round(fcc / 5))
    fcc_array.append(fcc)
    return fcc


def cadence_for_speed(v, t_dc_array, fcc_array):
    # print("VOOR: "+str(len(t_dc_array)))
    fcc = bicycle_model(t_dc_array, fcc_array)
    # print("NA: "+str(len(t_dc_array)))
    if v <= 15:
        rpm = v * 70 / 15
    else:
        rpm = 70 + (v - 15) * 8
    if rpm > fcc:
        return fcc
    return rpm


def fietsers_koppel(angle, t_dc, t_dc_array, t_cyclist_no_noise):
    gaussian_random = np.random.normal(0, 0.3)
    t_dc_noise = t_dc + gaussian_random
    if t_dc_noise < 0:
        t_dc_noise = 0
    t_dc_array.append(t_dc_noise)
    t_cyclist_no_noise.append(t_dc * (1 + sin(2 * angle - (pi / 6))))
    return t_dc_noise * (1 + sin(2 * angle - (pi / 6)))


def update(h, omega_crank, v_fiets):
    '''
    Updates the t_dc and t_dc_max values based on the speed and rpm at the given time. Also updates slope based on a perlin noise
    :param h: timestep
    '''
    t_dc_max = (-omega_crank[h - 1]) / 2 + 60
    t_dc = min(t_dc_max, max(0, -K * (v_fiets[h - 1] - v_fiets_ref)))
    n = noise.pnoise1(slope_offset + (h / 2000), 6, 0.1, 3, 1024)
    slope_rad = np.interp(n, [-1, 1], [-0.02, 0.07])
    return t_dc, t_dc_max, slope_rad
