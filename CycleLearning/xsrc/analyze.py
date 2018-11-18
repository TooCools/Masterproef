import matplotlib.pyplot as plt


def visualize_data(ys, legends, name="", path="", save=False):
    plt.title(name)
    for y in ys:
        plt.plot(y)
    plt.legend(legends)
    if save:
        plt.savefig(path + name)
        plt.close()
    else:
        plt.show()