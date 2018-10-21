from math import sin, pi, cos
import numpy as np
from numpy.fft import fft, fftfreq, ifft
from data_to_csv import save
from params import *
from route_slot import Slot
import matplotlib.pyplot as plt

theta_crank = [0.0]  # Hoek van van de trapas
omega_crank = [0.0]  # rpm van de trapas
v_fiets = [0.0]  # snelheid in m/s
t_cy = [0.0]
t_mg1 = [0.0]
t_mg2 = [0.0]
o_mg1 = [0.0]
o_mg2 = [0.0]
t_dc_array = [0.0]
slope_array = [0.0]
f_grav_array = [0.0]
f_fric_array = [0.0]
f_aero_array = [0.0]
start = True

rpm_offset = 0
t_dc = 50.0
slope = 0.0  # helling waarop de fiets zich bevindt
route_slots = []
total_timesteps = 10000

'''
Bepaald hoeveel vermogen een fietser trapt. P=T*O, met P het vermogen, T = t_dc en O= optimale cadans.
Een fietser trapt ongeveer 100W als hij stevig rechtdoor fietst, 150W als hij bergop fiets.
'''


def cadence_for_speed(v):
    """
    Berekent de cadans in rpm aan de hand van de snelheid
    :param v: snelheid in km/h
    :return: cadans in rpm
    """
    if start:
        if v <= 15:
            return v * 50 / 15
        return 50 + (v - 15) * 2
    else:
        if t_dc > 14.3:
            return 100
        if t_dc > 13.8:
            return 95
        elif t_dc > 13.2:
            return 90
        elif t_dc > 12.6:
            return 85
        elif t_dc > 11.9:
            return 80
        elif t_dc > 11.1:
            return 75
        elif t_dc > 10.2:
            return 70
        elif t_dc > 9.9:
            return 65
        elif t_dc > 9.5:
            return 60
        elif t_dc > 9.1:
            return 55
        elif t_dc > 8.55:
            return 50
        elif t_dc > 7.95:
            return 45
        else:
            return 40


def fietsers_koppel(angle):
    '''

    :param angle: An angle in radians
    :return: een waarde voor het fietserskoppel
    '''
    gaussian_random = np.random.normal(0, 0.5)
    t_dc_noise = t_dc + gaussian_random
    if t_dc_noise < 0:
        t_dc_noise = 0
    t_dc_array.append(t_dc_noise)
    return t_dc_noise * (1 + sin(2 * angle - (pi / 6)))


def get_tdc_from_rpm(rpm, watt):
    return watt / (rpm * 0.10467)


def init():
    route_slots.append(Slot(50, 10.2, 0, 0, 0, 20, 50))  # start cycling
    route_slots.append(Slot(10.2, 12, 0, 0.035, 5, 600, 630))  # change in slope 0°->2°
    route_slots.append(Slot(12, 15, 0.035, 0.18, 5, 1000, 1020))  # change in slope 2°->10°
    route_slots.append(Slot(15, 9.5, 0.18, -0.035, -15, 1200, 1280))  # change in slope 10°->-2°
    route_slots.append(Slot(9.5, 10.2, -0.035, 0, 5, 1400, 1410))  # change in slope -2°->0°
    route_slots.append(Slot(10.2, 12, 0, 0.035, 5, 3000, 3010))  # change in slope 0°->2°
    route_slots.append(Slot(12, 11.2, 0.035, 0.017, 0, 5000, 5050))  # change in slope 2°->1°
    route_slots.append(Slot(11.2, 0, 0.017, -0.035, -75, 5050, 5200))  # change in slope 1°->-4°
    route_slots.append(Slot(0, 11.1, -0.035, 0.017, 70, 5500, 5600))  # change in slope -4°->1°
    route_slots.append(Slot(11.1, 10.2, 0.017, 0, 0, 7000, 7100))  # change in slope 1°->0°
    # route_slots.append(Slot(10, 13, 5000, 5100))  # +- 75->100W
    # route_slots.append(Slot(13, 10, 8000, 8100))  # +- 100->75W
    # route_slots.append(Slot(10, 0, total_timesteps - 350, total_timesteps - 300))  # 75W-> 0W


def update(time):
    if len(route_slots) != 0:
        slot = route_slots[0]
        if not (slot.is_relevant(time)):
            route_slots.remove(slot)
        else:
            slot = route_slots[0]
            if slot.is_in_timeslot(time):
                global t_dc
                global slope
                t_dc += slot.get_change_t_dc()
                if t_dc < 0:
                    t_dc = 0
                slope += slot.get_change_slope()


def simulate():
    global start
    for h in range(1, total_timesteps):  # 1000s, 10 keer per seconde
        if start and h > 100:
            start = False
        update(h)

        v_fiets_previous_kmh = v_fiets[h - 1]
        v_fiets_previous_ms = v_fiets_previous_kmh / 3.6

        omega_crank_current_rpm = cadence_for_speed(
            v_fiets_previous_kmh)  # min(omega_opt_rpm, cadence_for_speed(v_fiets_previous_kmh)) + rpm_offset
        omega_crank_current_rads = omega_crank_current_rpm * 0.10467

        theta_crank_current_rad = theta_crank[h - 1] + omega_crank_current_rads * timestep

        # Dit zijn waarden voor het vermogen (torque) van de fietser + motor generatoren op het voor- (2) en achterwiel (1)
        t_cyclist = fietsers_koppel(theta_crank_current_rad)
        t_mg1_current = t_cyclist * kcr_r * (ns / nr) * ks_mg1
        t_mg2_current = min(20, support_level * t_cyclist)
        t_rw = t_cyclist * kcr_r * ((nr + ns) / nr)
        print('time', int(h / 10), 'speed', v_fiets_previous_kmh, 'slope', slope, 'rpm', omega_crank_current_rpm, 'tdc',
              t_dc_array[h])
        f_grav = total_mass * g * sin(slope) * 0.3
        f_friction = total_mass * g * cos(slope) * cr
        f_aero = 0.5 * cd * ro_aero * a_aero * (v_fiets_previous_ms ** 2)
        f_load = f_grav + f_friction + f_aero
        f_load *= np.sign(v_fiets_previous_kmh)

        v_fiets_next_ms = (((t_mg2_current + t_rw) / rw) - f_load) / total_mass
        v_fiets_current_ms = v_fiets_previous_ms + timestep * v_fiets_next_ms

        omega_mg2 = v_fiets_current_ms / rw
        omega_mg1 = (1 / ks_mg1) * ((1 + (nr / ns)) * omega_mg2 - ((nr / ns) * (omega_crank_current_rads / kcr_r)))

        theta_crank.append(theta_crank_current_rad)
        omega_crank.append(omega_crank_current_rpm)
        v_fiets.append(v_fiets_current_ms * 3.6)
        t_cy.append(t_cyclist)
        t_mg1.append(t_mg1_current)
        t_mg2.append(t_mg2_current)
        o_mg1.append(omega_mg1)
        o_mg2.append(omega_mg2)
        slope_array.append(slope * 57.296)
        f_grav_array.append(f_grav)
        f_fric_array.append(f_friction)
        f_aero_array.append(f_aero)


init()
simulate()
data = {'speed (km/h)': v_fiets,
        'rpm': omega_crank,
        'crank angle': theta_crank,
        't_cyclist': t_cy,
        't_mg1': t_mg1,
        't_mg2': t_mg2,
        'o_mg1': o_mg1,
        'o_mg2': o_mg2,
        't_dc': t_dc_array,
        'slope °': slope_array,
        'force gravity': f_grav_array,
        'force friction': f_fric_array,
        'force aero': f_aero_array
        }

save(data)


def fourier():
    size = 100
    start_index = 2000
    x = theta_crank[start_index:start_index + size]
    y = t_cy[start_index:start_index + size]
    plt.figure(1)
    plt.title("Original function")
    plt.plot(x, y)
    freqs = fftfreq(size, 0.115)
    fft_vals = fft(y)
    fft_theo = 2 * np.abs(fft_vals / size)
    mask = freqs > 0
    actual_freqs = freqs[mask]
    actual_fft = fft_theo[mask]
    plt.figure(2)
    plt.plot(actual_freqs, actual_fft)
    plt.show()
    print('peek', actual_fft[np.argmax(actual_fft)])
    t_dc_sub = t_dc_array[start_index:start_index + size]
    print('average t_dc', np.average(t_dc_sub))


# fourier()
print("Finished")
