from matplotlib import pyplot as plt

def plot_dist(deg, cnt, color='b', title='Degree Distribution'):
    plt.plot(deg, cnt)
    plt.title(title)
    plt.ylabel("Count")
    plt.xlabel("Degree")
    # plt.show()