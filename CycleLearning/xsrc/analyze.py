import matplotlib.pyplot as plt


def visualize_data(ys, legends, assen, name="", path="", save=False):
    plt.title(name)
    for y in ys:
        plt.plot(y)
    plt.legend(legends)
    plt.xlabel(assen[0])
    plt.ylabel(assen[1])
    if save:
        plt.savefig(path + name+".png")
        plt.show()
        plt.close()
    else:
        plt.show()
