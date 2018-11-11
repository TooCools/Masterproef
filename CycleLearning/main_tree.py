from excel_to_data import get_data
from learner import learn_regtree, learn_randomforest
from params import *
from preprocessing import preprocess
import time
import matplotlib.pyplot as plt

# df_slope = 'slope °'

# df_main = get_data("..\\Fietssimulatie\\data.xlsx", [df_torque, df_crank_angle_rad, df_rpm])
# df_val = get_data("..\\Fietssimulatie\\validation.xlsx", [df_torque, df_crank_angle_rad, df_rpm])

print("Loading Data")
df_main = get_data("data\\data.xlsx", [df_torque, df_crank_angle_rad, df_rpm])
df_val = get_data("data\\validation.xlsx", [df_torque, df_crank_angle_rad, df_rpm])
print("Data loaded")


def visualize_data(ys, legends, name, path):
    plt.title(name)
    for y in ys:
        plt.plot(y)
    plt.legend(legends)
    plt.savefig(path + name)
    plt.close()


for seq_len in range(10, 101, 10):
    print("Preprocessing data")
    train_x, train_y = preprocess(df_main, seq_len, normalize=False)
    val_x, val_y = preprocess(df_val, seq_len, normalize=False, shuffle=False)
    nsamples, nx, ny = train_x.shape
    d2_train_dataset = train_x.reshape((nsamples, nx * ny))
    nsamples2, nx2, ny2 = val_x.shape
    d2_val_dataset = val_x.reshape((nsamples2, nx2 * ny2))

    print("Start learning trees with seqlen " + str(seq_len))
    for tree_depth in range(8, 12):
        name = f"TREE_SEQ_{seq_len}_DEPTH_{tree_depth}_T_{int(time.time())}"
        tree = learn_regtree(name, train_x, train_y, tree_depth)
        predictions = tree.predict(d2_val_dataset)
        visualize_data([predictions, val_y], ["predictions", "actual"], name, "models/trees/")

    print("Start learning forests with seqlen " + str(seq_len))
    for tree_depth in range(8, 12):
        for estimators in range(8, 12):
            name = f"FOREST_SEQ_{seq_len}_DEPTH_{tree_depth}_ESTIMATORS_{estimators}_T_{int(time.time())}"
            forest = learn_randomforest(name, train_x, train_y, estimators, tree_depth)
            predictions = forest.predict(d2_val_dataset)
            visualize_data([predictions, val_y], ["predictions", "actual"], name, "models/forests/")


