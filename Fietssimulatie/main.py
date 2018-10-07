from math import sin, pi
from params import *

theta_crank = [0]  # Hoek van van de trapas
omega_crank = [0]  # rpm van de trapas
v_fiets = [0]  # snelheid in m/s

t_dc=1





def f(v):
    """
    Berekent de cadans? aan de hand van de snelheid
    :param v: snelheid in km/h
    :return: cadans in rpm
    """
    if v <= 15:
        return v * 50 / 15
    return 50 + (v - 15) * 2

phi=pi/6
def g(theta):
    return t_dc*(1+sin(2*theta-pi))


for h in range(1, 10000):  # 1000s, 10 keer per seconde
    v_fiets_prev_kmh = v_fiets[h - 1] * 3.6
    omega_crank_current_rpm = min(omega_opt_rpm, f(v_fiets_prev_kmh))
    omega_crank_current_rads = omega_crank_current_rpm * 0.10467
    theta_crank_current_rad = theta_crank[h - 1] + omega_crank_current_rads * timestep

    # Wat betekenen deze t_ waarden?
    t_cyclist = g(theta_crank_current_rad)
    t_mg1 = t_cyclist * kcr_r * (ns / nr) * ks_mg1
    t_mg2 = min(25, 5 * t_cyclist)  # is dit 5 of s? als het s is, wat is s?

    t_rw = t_cyclist * kcr_r * ((nr + ns) / nr)
    v_fiets_current = v_fiets[h - 1] + timestep * (1 / total_mass) * ((t_mg2 + t_rw) / rw)

    theta_crank.append(theta_crank_current_rad)
    omega_crank.append(omega_crank_current_rpm)
    v_fiets.append(v_fiets_current)

    omega_mg2 = v_fiets_current / rw
    omega_mg1 = (1 / ks_mg1) * ((1 + (nr / ns)) * omega_mg2 * ((nr / ns) * (omega_crank_current_rads / kcr_r)))

print("Finished")
