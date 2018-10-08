import pandas as pd


def save(km_h, rpm, cr_angle, t_cy, t_mg1, t_mg2, o_mg1, o_mg2):
    data = {'speed (km/h)': km_h,
            'rpm': rpm,
            'crank angle': cr_angle,
            't_cyclist': t_cy,
            't_mg1': t_mg1,
            't_mg2': t_mg2,
            'o_mg1': o_mg1,
            'o_mg2': o_mg2
            }
    df = pd.DataFrame(data,
                      columns=['speed (km/h)', 'rpm', 'crank angle', 't_cyclist', 't_mg1', 't_mg2', 'o_mg1', 'o_mg2'])
    df.to_csv('data.csv', index=False, decimal=',')
