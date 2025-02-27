kcr_r = 21 / 52  # Verhouding overbrenging trapas-ringwiel
ks_mg1 = 30 / 113  # Verhouding overbrenging zon-MG1
ns = 28  # Aantal tanden zonnewiel
nr = 83  # Aantal tanden ringwiel
rw = 0.3231  # Straal van voor- en achterwiel
total_mass = 100  # gewicht fietser + fiets in kg
timestep = .1  # tijdsverschil tussen 2 berekeningen in seconde
support_level = 2  # Ondersteuningsniveau

g = 9.81  # zwaartekracht constante
cr = 0.004  # rolling friction coefficient
cd = 1.05  # aerodynamic drag coefficient
ro_aero = 1.2  # air density kg/m³
a_aero = 0.5  # frontal area of cyclist and bicycle m²

K = 6  # hoe agressief de fietser zijn snelheid zoekt te behalen (zie proportionele regelaar)

seqlen = 50  # gekozen sequentie lengte

# namen van kolommen
df_torque = 't_cyclist'
df_crank_angle_rad = 'crank_angle_%2PI'
df_fcc = 'fcc'
df_velocity = "speed (km/h)"
df_slope = "slope °"
