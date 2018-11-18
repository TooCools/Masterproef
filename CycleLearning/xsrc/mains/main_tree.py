from xsrc.analyze import visualize_data
from xsrc.excel_to_data import get_data
from xsrc.learner import learn_randomforest
from xsrc.params import *
from xsrc.preprocessing import preprocess
# df_slope = 'slope °'

# df_main = get_data("..\\Fietssimulatie\\data.xlsx", [df_torque, df_crank_angle_rad, df_rpm])
# df_val = get_data("..\\Fietssimulatie\\validation.xlsx", [df_torque, df_crank_angle_rad, df_rpm])

df_main = get_data("..\\data\\data.xlsx", [df_torque, df_crank_angle_rad, df_fcc])
df_val = get_data("..\\data\\validation.xlsx", [df_torque, df_crank_angle_rad, df_fcc])




# val_x, val_y = preprocess(df_val, 50, normalize=False, shuffle=False, classification=True)

train_x, train_y = preprocess(df_main, 50, normalize=False, classification=True)
val_x, val_y = preprocess(df_val, 50, normalize=False, shuffle=False, classification=True)
nsamples, nx, ny = train_x.shape
d2_train_dataset = train_x.reshape((nsamples, nx * ny))
nsamples2, nx2, ny2 = val_x.shape
d2_val_dataset = val_x.reshape((nsamples2, nx2 * ny2))
forest = learn_randomforest("", train_x, train_y, 5,9, classification=True)
predictions = forest.predict(d2_val_dataset)

for (i, item) in enumerate(predictions):
    predictions[i] = (40 + item * 5)

for (i, item) in enumerate(val_y):
    val_y[i] = (40 + item * 5)

visualize_data([predictions, val_y], ["predictions", "actual"], "", "models/forests/")

# for seq_len in range(10, 101, 10):
#     print("Preprocessing data")
#     train_x, train_y = preprocess(df_main, seq_len, normalize=False)
#     val_x, val_y = preprocess(df_val, seq_len, normalize=False, shuffle=False)
#     nsamples, nx, ny = train_x.shape
#     d2_train_dataset = train_x.reshape((nsamples, nx * ny))
#     nsamples2, nx2, ny2 = val_x.shape
#     d2_val_dataset = val_x.reshape((nsamples2, nx2 * ny2))
#
#     print("Start learning trees with seqlen " + str(seq_len))
#     for tree_depth in range(8, 12):
#         name = f"TREE_SEQ_{seq_len}_DEPTH_{tree_depth}_T_{int(time.time())}"
#         tree = learn_regtree(name, train_x, train_y, tree_depth)
#         predictions = tree.predict(d2_val_dataset)
#         visualize_data([predictions, val_y], ["predictions", "actual"], name, "models/trees/")
#
#     print("Start learning forests with seqlen " + str(seq_len))
#     for tree_depth in range(8, 12):
#         for estimators in range(8, 12):
#             name = f"FOREST_SEQ_{seq_len}_DEPTH_{tree_depth}_ESTIMATORS_{estimators}_T_{int(time.time())}"
#             forest = learn_randomforest(name, train_x, train_y, estimators, tree_depth)
#             predictions = forest.predict(d2_val_dataset)
#             visualize_data([predictions, val_y], ["predictions", "actual"], name, "models/forests/")
