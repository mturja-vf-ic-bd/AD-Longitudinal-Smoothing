from matplotlib import pyplot as plt

def plot_dist(deg, cnt, color='b', title='Degree Distribution', xlabel='deg', ylabel='cnt'):
    plt.plot(deg, cnt)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    # plt.show()