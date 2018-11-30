from math import sin, pi, cos
import matplotlib.pyplot as plt
import noise
import numpy as np
from data_to_csv import save
from params import *

theta_crank_rad = [0.0]  # Hoek van van de trapas
theta_crank_rad2 = [0.0]  # Hoek van de trapas %2PI
omega_crank = [0.0]  # rpm van de trapas
v_fiets = [0.0]  # snelheid in m/s
t_cy = [0.0]
t_mg1 = [0.0]
t_mg2 = [0.0]
o_mg1 = [0.0]
o_mg2 = [0.0]
t_dc_array = [0.0]
t_cyclist_no_noise = [0.0]
slope_array = [0.0]
f_grav_array = [0.0]
f_fric_array = [0.0]
f_aero_array = [0.0]
fcc_array = [0.0]
t_dc_max = 60
slope_offset = 19

t_dc = 0.0
slope_rad = 0.0  # helling waarop de fiets zich bevindt
route_slots = []
total_timesteps = 100000

'''
Bepaald hoeveel vermogen een fietser trapt. P=T*O, met P het vermogen, T = t_dc en O= optimale cadans.
Een fietser trapt ongeveer 100W als hij stevig rechtdoor fietst, 150W als hij bergop fiets.
'''


def bicycle_model():
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


def cadence_for_speed(v):
    """
    Calculates the cadence for a given speed when accelerating, otherwise based on torque
    :param v: speed in km/h
    :return: cadence in rpm
    """
    fcc = bicycle_model()
    if v <= 15:
        rpm = v * 70 / 15
    else:
        rpm = 70 + (v - 15) * 8
    if rpm > fcc:
        return fcc
    return rpm


def fietsers_koppel(angle):
    '''
    :param angle: An angle in radians
    :return: returns a simulated value of the cyclers torque
    '''
    gaussian_random = np.random.normal(0, 0.3)
    t_dc_noise = t_dc + gaussian_random
    if t_dc_noise < 0:
        t_dc_noise = 0
    t_dc_array.append(t_dc_noise)
    t_cyclist_no_noise.append(t_dc * (1 + sin(2 * angle - (pi / 6))))
    return t_dc_noise * (1 + sin(2 * angle - (pi / 6)))


def update(h):
    '''
    Updates the t_dc and t_dc_max values based on the speed and rpm at the given time. Also updates slope based on a perlin noise
    :param h: timestep
    '''
    global t_dc
    global t_dc_max
    t_dc_max = (-omega_crank[h - 1]) / 2 + 60
    t_dc = min(t_dc_max, max(0, -K * (v_fiets[h - 1] - v_fiets_ref)))
    global slope_rad
    n = noise.pnoise1(slope_offset + (h / 2000), 6, 0.1, 3, 1024)
    slope_rad = np.interp(n, [-1, 1], [-0.02, 0.07])


def simulate():
    for h in range(1, int(total_timesteps)):
        update(h)
        v_fiets_previous_kmh = v_fiets[h - 1]
        v_fiets_previous_ms = v_fiets_previous_kmh / 3.6

        omega_crank_current_rpm = cadence_for_speed(
            v_fiets_previous_kmh)  # min(omega_opt_rpm, cadence_for_speed(v_fiets_previous_kmh)) + rpm_offset
        omega_crank_current_rads = omega_crank_current_rpm * 0.10467

        theta_crank_current_rad = theta_crank_rad[h - 1] + omega_crank_current_rads * timestep

        # Dit zijn waarden voor het vermogen (torque) van de fietser + motor generatoren op het voor- (2) en achterwiel (1)
        t_cyclist = fietsers_koppel(theta_crank_current_rad)
        t_mg1_current = t_cyclist * kcr_r * (ns / nr) * ks_mg1
        t_mg2_current = min(20, support_level * t_cyclist)
        t_rw = t_cyclist * kcr_r * ((nr + ns) / nr)
        print('time', int(h / 10), 'speed', v_fiets_previous_kmh, 'slope', slope_rad, 'rpm', omega_crank_current_rpm,
              'tdc',
              t_dc_array[h])
        f_grav = total_mass * g * sin(slope_rad)*0.9
        f_friction = total_mass * g * cos(slope_rad) * cr
        f_aero = 0.5 * cd * ro_aero * a_aero * (v_fiets_previous_ms ** 2)
        f_aero *= np.sign(v_fiets_previous_kmh)
        f_load = f_grav + f_friction + f_aero

        v_fiets_next_ms = (((t_mg2_current + t_rw) / rw) - f_load) / total_mass
        v_fiets_current_ms = v_fiets_previous_ms + timestep * v_fiets_next_ms

        omega_mg2 = v_fiets_current_ms / rw
        omega_mg1 = (1 / ks_mg1) * ((1 + (nr / ns)) * omega_mg2 - ((nr / ns) * (omega_crank_current_rads / kcr_r)))

        theta_crank_rad.append(theta_crank_current_rad)
        theta_crank_rad2.append(theta_crank_current_rad % (2 * pi))
        omega_crank.append(omega_crank_current_rpm)
        v_fiets.append(v_fiets_current_ms * 3.6)
        t_cy.append(t_cyclist)
        t_mg1.append(t_mg1_current)
        t_mg2.append(t_mg2_current)
        o_mg1.append(omega_mg1)
        o_mg2.append(omega_mg2)
        slope_array.append(slope_rad * 57.296)
        f_grav_array.append(f_grav)
        f_fric_array.append(f_friction)
        f_aero_array.append(f_aero)


# init()
simulate()
data = {'speed (km/h)': v_fiets,
        'rpm': omega_crank,
        'crank_angle': theta_crank_rad,
        'crank_angle_%2PI': theta_crank_rad2,
        't_dc': t_dc_array,
        't_cyclist': t_cy,
        't_cyclist_no_noise': t_cyclist_no_noise,
        't_mg1': t_mg1,
        't_mg2': t_mg2,
        'o_mg1': o_mg1,
        'o_mg2': o_mg2,
        'slope °': slope_array,
        'force gravity': f_grav_array,
        'force friction': f_fric_array,
        'force aero': f_aero_array,
        'fcc': fcc_array
        }

# save(data)
## save(data, "validation")
# print("Finished")
#
#
# def visualize_data(y):
#     for data in y:
#         plt.plot(data)
#     plt.show()
#
#
# visualize_data([omega_crank, slope_array])
