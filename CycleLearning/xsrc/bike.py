from math import sin, pi, cos

import noise
import numpy as np

from xsrc.simulation.data_to_csv import save
from xsrc.params import timestep, kcr_r, support_level, ns, nr, ks_mg1, total_mass, g, \
    ro_aero, cr, cd, a_aero, K, rw


class Bike:
    crank_angle_rad = [0.0]  # Hoek van van de trapas
    crank_speed_rpm = [0.0]  # rpm van de trapas
    v_fiets_kmh = [0.0]  # snelheid in m/s
    t_cy = [0.0]
    t_mg1 = [0.0]
    t_mg2 = [0.0]
    t_rw = [0.0]
    o_mg1 = [0.0]
    o_mg2 = [0.0]
    t_dc_array = [0.0]
    t_cyclist_no_noise = [0.0]
    slope_array = [0.0]
    fcc_array = [0.0]
    v_fiets_ref = 32
    t_dc_max = 60
    t_dc = 0.0

    # slope_rad = 0.0  # helling waarop de fiets zich bevindt

    def __init__(self, cycle_model=(lambda cm_tdc, cm_speed, cm_slope: 7.5 * cm_tdc), verbose=False):
        self.slope_offset = 19
        self.dominant_leg = False
        self.f_load = 0
        self.verbose = verbose
        self.cycle_model = cycle_model

    def get_recent_data(self, h, amount):
        if len(self.t_cy) < amount:
            zeros = np.zeros(amount - h - 1)
            return np.append(zeros, self.t_cy), np.append(zeros, self.crank_angle_rad), \
                   np.append(zeros, self.v_fiets_kmh), np.append(zeros, self.slope_array)
        else:
            return self.t_cy[-amount:], self.crank_angle_rad[-amount:], self.v_fiets_kmh[-amount:], self.slope_array[
                                                                                                    -amount:]

    def get_recent_fcc(self, h, amount):
        if len(self.fcc_array) < amount:
            return np.append(np.zeros(amount - h - 1), self.fcc_array)
        else:
            return self.fcc_array[-amount:]

    def update(self, h, prediction=-1):
        self.update_torque()
        self.update_slope(h)
        self.update_cadence(prediction)
        self.update_cycle_torque()
        self.update_load()
        self.update_speed()
        if self.verbose:
            print('time', int(h / 10), 'speed', self.v_fiets_kmh[-1], 'slope', self.slope_array[-1], 'rpm',
                  self.crank_speed_rpm[-1],
                  'tdc',
                  self.t_dc)

    def update_torque(self):
        self.t_dc_max = (-self.crank_speed_rpm[-1]) / 2 + 60
        self.t_dc = min(self.t_dc_max, max(0, -K * (self.v_fiets_kmh[-1] - self.v_fiets_ref)))

    def update_slope(self, h):
        n = noise.pnoise1(self.slope_offset + (h / 2000), 6, 0.1, 3, 1024)
        slope = np.interp(n, [-1, 1], [0, 0.1])
        self.slope_array.append(slope)

    def update_cadence(self, fcc):
        temp = self.get_fcc()
        if fcc == -1:
            fcc = temp
        v_kmh = self.v_fiets_kmh[-1]
        if v_kmh <= 15:
            rpm = v_kmh * 50 / 15
        else:
            rpm = 50 + (v_kmh - 15) * 2

        if rpm > fcc:
            # print("Going FCC baby")
            rpm = fcc
        # else:
            # print("@@@@@@@@@@@@@@@@@@")
        self.crank_speed_rpm.append(rpm)

    def update_cycle_torque(self):
        crank_speed_rads = self.crank_speed_rpm[-1] * 0.10467
        theta_crank_current_rad = self.crank_angle_rad[-1] + crank_speed_rads * timestep
        torque = self.cycler_torque(theta_crank_current_rad)
        self.t_cy.append(torque)
        # self.t_mg1.append(torque * kcr_r * (ns / nr) * ks_mg1)
        self.t_mg2.append(min(35, support_level * torque))
        self.t_rw.append(torque * kcr_r * ((nr + ns) / nr))
        self.crank_angle_rad.append(theta_crank_current_rad % (2 * pi))

    def update_load(self):
        v_fiets_ms = self.v_fiets_kmh[-1] / 3.6
        f_grav = total_mass * g * sin(self.slope_array[-1])
        f_friction = total_mass * g * cos(self.slope_array[-1]) * cr
        f_aero = 0.5 * cd * ro_aero * a_aero * (v_fiets_ms ** 2)
        f_aero *= np.sign(v_fiets_ms)
        self.f_load = f_grav + f_friction + f_aero

    def update_speed(self):
        v_fiets_next_ms = (((self.t_mg2[-1] + self.t_rw[-1]) / rw) - self.f_load) / total_mass
        v_fiets_previous_ms = self.v_fiets_kmh[-1] / 3.6
        v_fiets_current_ms = v_fiets_previous_ms + timestep * v_fiets_next_ms
        self.v_fiets_kmh.append(v_fiets_current_ms * 3.6)
        omega_mg2 = v_fiets_current_ms / rw
        omega_crank_current_rads = self.crank_speed_rpm[-1] * 0.10467
        omega_mg1 = (1 / ks_mg1) * ((1 + (nr / ns)) * omega_mg2 - ((nr / ns) * (omega_crank_current_rads / kcr_r)))
        # self.o_mg1.append(omega_mg1)
        # self.o_mg2.append(omega_mg2)

    def cycler_torque(self, angle):
        gaussian_random = np.random.normal(0, 0.3) * 5
        t_dc_noise = self.t_dc + gaussian_random
        if t_dc_noise < 0:
            t_dc_noise = 0
        self.t_dc_array.append(t_dc_noise)
        self.t_cyclist_no_noise.append(self.t_dc * (1 + sin(2 * angle - (pi / 6))))

        if not self.dominant_leg:
            torque = t_dc_noise * (1 + sin(2 * angle - (pi / 6)))
        else:
            torque = t_dc_noise * (1 + sin(2 * angle - (pi / 6)) + sin(angle - (pi / 3)) / 2.5)
        return torque

    def get_fcc(self):
        fcc = self.cycle_model(self.t_dc, self.v_fiets_kmh[-1], self.slope_array[-1])
        if fcc < 40:
            fcc = 40
        elif fcc > 120:
            fcc = 120
        self.fcc_array.append(fcc)
        return fcc

    def save(self):
        data = {'speed (km/h)': self.v_fiets_kmh,
                'rpm': self.crank_speed_rpm,
                'crank_angle_%2PI': self.crank_angle_rad,
                't_dc': self.t_dc_array,
                't_cyclist': self.t_cy,
                't_cyclist_no_noise': self.t_cyclist_no_noise,
                't_mg1': self.t_mg1,
                't_mg2': self.t_mg2,
                'o_mg1': self.o_mg1,
                'o_mg2': self.o_mg2,
                'slope °': self.slope_array,
                'total load': self.f_load,
                'fcc': self.fcc_array
                }
        save(data, "test")
