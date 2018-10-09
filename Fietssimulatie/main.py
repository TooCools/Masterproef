from math import sin, pi

from data_to_csv import save
from params import *

theta_crank = [0]  # Hoek van van de trapas
omega_crank = [0]  # rpm van de trapas
v_fiets = [0]  # snelheid in m/s
t_cy = [0]
t_mg1 = [0]
t_mg2 = [0]
o_mg1 = [0]
o_mg2 = [0]

t_dc = 1.429
'''
Bepaald hoeveel vermogen een fietser trapt. P=T*O, met P het vermogen, T = t_dc en O= optimale cadans.
Een fietser trapt ongeveer 100W als hij stevig rechtdoor fietst, 150W als hij bergop fiets. Dus T=P/O => T=1.429 vlak en T=2.143
'''


def f(v):
    """
    Berekent de cadans in rpm aan de hand van de snelheid
    :param v: snelheid in km/h
    :return: cadans in rpm
    """
    if v <= 15:
        return v * 50 / 15
    return 50 + (v - 15) * 2


def g(angle):
    '''

    :param angle: An angle in radians
    :return: een waarde voor het fietserskoppel
    '''
    return t_dc * (1 + sin(2 * angle - (pi / 6)))


for h in range(1, 10000):  # 1000s, 10 keer per seconde
    omega_crank_current_rpm = min(omega_opt_rpm, f(v_fiets[h - 1]))
    omega_crank_current_rads = omega_crank_current_rpm * 0.10467

    theta_crank_current_rad = theta_crank[h - 1] + omega_crank_current_rads * timestep

    # Dit zijn waarden voor het vermogen van de fietser, en beide motor generatoren op het voor- (2) en achterwiel (1)
    t_cyclist = g(theta_crank_current_rad)
    t_mg1_current = t_cyclist * kcr_r * (ns / nr) * ks_mg1
    t_mg2_current  = min(25, support_level * t_cyclist)  # is dit 5 of s? als het s is, wat is s?

    t_rw = t_cyclist * kcr_r * ((nr + ns) / nr)
    v_fiets_current = v_fiets[h - 1] / 3.6 + timestep * (1 / total_mass) * ((t_mg2_current + t_rw) / rw)

    omega_mg2 = v_fiets_current / rw
    omega_mg1 = (1 / ks_mg1) * ((1 + (nr / ns)) * omega_mg2 - ((nr / ns) * (omega_crank_current_rads / kcr_r)))

    theta_crank.append(theta_crank_current_rad)
    omega_crank.append(omega_crank_current_rpm)
    v_fiets.append(v_fiets_current * 3.6)
    t_cy.append(t_cyclist)
    t_mg1.append(t_mg1_current)
    t_mg2.append(t_mg2_current)
    o_mg1.append(omega_mg1)
    o_mg2.append(omega_mg2)

save(v_fiets, omega_crank, theta_crank, t_cy, t_mg1, t_mg2, o_mg1, o_mg2)
print("Finished")
