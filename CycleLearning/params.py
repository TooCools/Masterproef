import time

SEQ_LEN = 50
EPOCH = 10
BATCH_SIZE = 64
NAME = f"SEQ_{SEQ_LEN}_EPOCH_{EPOCH}_{int(time.time())}"

df_torque = 't_cyclist'
df_crank_angle_rad = 'crank_angle_%2PI'
df_rpm = 'rpm'