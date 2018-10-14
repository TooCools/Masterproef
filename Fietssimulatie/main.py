from math import sin, pi, cos

from data_to_csv import save
from params import *
from route_slot import Slot

theta_crank = [0]  # Hoek van van de trapas
omega_crank = [0]  # rpm van de trapas
v_fiets = [0]  # snelheid in m/s
t_cy = [0]
t_mg1 = [0]
t_mg2 = [0]
o_mg1 = [0]
o_mg2 = [0]
t_dc_array = [0]

t_dc = 0
route_slots = []

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
    if v <= 15:
        return v * 50 / 15
    return 50 + (v - 15) * 2


def fietsers_koppel(angle):
    '''

    :param angle: An angle in radians
    :return: een waarde voor het fietserskoppel
    '''
    return t_dc * (1 + sin(2 * angle - (pi / 6)))


def get_tdc_from_rpm(rpm, watt):
    return watt / (rpm * 0.10467)


def init():
    # route_slots.append(Slot(0, 1.429, 0, 50))  # start cycling 0W->100W
    # route_slots.append(Slot(1.429, 2.1428, 5000, 5100))  # 100W->150W
    # route_slots.append(Slot(2.1428, 1.429, 8000, 8100))  # 150W->100W
    # route_slots.append(Slot(1.429, 0, 9900, 10000))  # 100->0W
    route_slots.append(Slot(0, 10, 0, 100))  # start cycling 0W->100W
    route_slots.append(Slot(10, 13, 5000, 5100))  # 100W->150W
    route_slots.append(Slot(13, 10, 8000, 8100))  # 150W->100W
    route_slots.append(Slot(10, 0, 9900, 10000))  # 100->0W


def update_t_dc(time):
    slot = route_slots[0]
    if not (slot.is_relevant(time)):
        route_slots.remove(slot)
    else:
        slot = route_slots[0]
        if slot.is_in_timeslot(time):
            global t_dc
            t_dc += slot.get_change_t_dc()


def simulate():
    for h in range(1, 10000):  # 1000s, 10 keer per seconde
        update_t_dc(h)

        v_fiets_previous_kmh = v_fiets[h - 1]
        v_fiets_previous_ms = v_fiets_previous_kmh / 3.6

        omega_crank_current_rpm = min(omega_opt_rpm, cadence_for_speed(v_fiets_previous_kmh))
        omega_crank_current_rads = omega_crank_current_rpm * 0.10467

        theta_crank_current_rad = theta_crank[h - 1] + omega_crank_current_rads * timestep

        # Dit zijn waarden voor het vermogen (torque) van de fietser + motor generatoren op het voor- (2) en achterwiel (1)
        t_cyclist = fietsers_koppel(theta_crank_current_rad)
        t_mg1_current = t_cyclist * kcr_r * (ns / nr) * ks_mg1
        t_mg2_current = min(25, support_level * t_cyclist)
        t_rw = t_cyclist * kcr_r * ((nr + ns) / nr)

        f_grav = total_mass * g * sin(slope)
        f_friction = total_mass * g * cos(slope) * cr
        f_aero = 0.5 * cd * ro_aero * a_aero * (v_fiets_previous_ms ** 2)
        f_load = f_grav + f_friction + f_aero

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
        t_dc_array.append(t_dc)


init()
#simulate()
#save(v_fiets, omega_crank, theta_crank, t_cy, t_mg1, t_mg2, o_mg1, o_mg2, t_dc_array)
print("Finished")
