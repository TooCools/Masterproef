from excel_to_data import get_data

from learner import learn_regtree, learn_randomforest
from params import *
from preprocessing import preprocess
import time

# df_slope = 'slope °'

df_main = get_data("..\\Fietssimulatie\\data.xlsx", [df_torque, df_crank_angle_rad, df_rpm])
df_val = get_data("..\\Fietssimulatie\\validation.xlsx", [df_torque, df_crank_angle_rad, df_rpm])

print("Loading Data")
# df_main = get_data("data\\data.xlsx", [df_torque, df_crank_angle_rad, df_rpm])
# df_val = get_data("data\\validation.xlsx", [df_torque, df_crank_angle_rad, df_rpm])
print("Data loaded")

print("Preprocessing data")
train_x, train_y = preprocess(df_main, SEQ_LEN_TREE, normalize=False)
val_x, val_y = preprocess(df_val, SEQ_LEN_TREE, normalize=False, shuffle=False)
nsamples, nx, ny = train_x.shape
d2_train_dataset = train_x.reshape((nsamples, nx * ny))

print("Start learning")
depth = 50
name = f"SEQ_{SEQ_LEN_TREE}_DEPTH{depth}_{int(time.time())}"
tree = learn_regtree(name, train_x, train_y, depth)
score_tree = tree.score(d2_train_dataset, val_y)
estimators = 20
forest = learn_randomforest(name, train_x, train_y, estimators, depth)
score_forest = forest.score(d2_train_dataset, val_y)
print("score tree ", score_tree)
print("score forest ", score_forest)
