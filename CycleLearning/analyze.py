import matplotlib.pyplot as plt


def visualize_data(ys, legends, axis, name="", path="", save=False):
    '''
    Creates and show a graph
    :param ys: Y-values to show
    :param legends: Legends of the Y-values (array of strings)
    :param axis: Name of X and Y axis (array of strings)
    :param name: Title of graph
    :param path: path to save graph to
    :param save: to save or not to save
    '''
    plt.title(name)
    for y in ys:
        plt.plot(y)
    plt.legend(legends)
    plt.xlabel(axis[0])
    plt.ylabel(axis[1])
    if save:
        plt.savefig(path + name + ".png")
        plt.show()
        plt.close()
    else:
        plt.show()
