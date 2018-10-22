from excel_to_data import get_data
from noise_remover import fourier

df_cy = 't_cyclist'
df_cr = 'crank_angle'
df_slope = 'slope °'
df_cy_nn = 't_cyclist_no_noise'

df = get_data([df_cy, df_cy_nn, df_slope, df_cr])
t_cyclist = df[df_cy]
t_cyclist_no_noise = df[df_cy_nn]
slope = df[df_slope]
crank_angle = df[df_cr]

fourier(t_cyclist, t_cyclist_no_noise)
