from Longitudinal_Classifier.helper import *
from numpy.polynomial.polynomial import polyfit
from matplotlib import pyplot as plt
import timeit


class cortical_change:
    def read_data(self, sub_name):
        self.sub = sub_name
        self.data = {}
        self.net = {}
        for sub in sub_name:
            d = read_subject_data(sub, conv_to_tensor=False)
            self.data[sub] = d
            net = get_aggr_net([d], reduce='mean')
            net = (net + net.T)//2
            self.net[sub] = net

    def process_data(self, normalize=False):
        self.node_feat = {}
        for name, sub in self.data.items():
            F = np.stack(sub['node_feature'], axis=1)
            if normalize:
                mu = F.mean(axis=0, keepdims=True)
                sig = F.std(axis=0, keepdims=True)
                F = (F - mu) / sig
            self.node_feat[name] = F

    def compute_node_slope(self):
        if len(self.node_feat) == 0:
            self.c = None
            self.m = None
            return
        self.c = {}
        self.m = {}
        for i, f in self.node_feat.items():
            c, m = self.get_gradient(f.T)
            self.c[i] = c
            self.m[i] = m

    def get_gradient(self, cort_list):
        # Fit a line to estimate longitudinal rate of change for a region
        c, m = polyfit(np.arange(0, len(cort_list)), np.array(cort_list), deg=1)
        return c, m

    def plot_fit(self, sub, reg):
        cort_list = self.node_feat[sub][reg, :]
        m = self.m[sub, reg]
        c = self.c[sub, reg]
        x = np.arange(0, len(cort_list))
        plt.plot(x, cort_list, label="Raw")
        self.abline(m, c, "Fitted")
        plt.legend()

    def abline(self, slope, intercept, label):
        """Plot a line from slope and intercept"""
        axes = plt.gca()
        x_vals = np.array(axes.get_xlim())
        y_vals = intercept + slope * x_vals
        plt.plot(x_vals, y_vals, '--', label=label)


if __name__ == '__main__':
    plt.figure(figsize=(18, 6))
    sub_names = ["005_S_5038"]
    cc = cortical_change()
    cc.read_data(sub_names)
    cc.process_data()
    start = timeit.default_timer()
    cc.compute_node_slope()
    end = timeit.default_timer()
    # np.savetxt(os.path.join(Args.HOME_DIR, 'slope_dx_' + str(i) +'.txt'), cc.m)
    # np.savetxt(os.path.join(Args.HOME_DIR, 'intercept_dx_' + str(i) +'.txt'), cc.c)
    print("RunTime: ", end - start)
    for r in range(148):
        for j in range(len(cc.m)):
            cc.plot_fit(j, r)
            # directory = Args.HOME_DIR + '/Longitudinal_Classifier' + '/_dx_' + str(i)
            # if not os.path.exists(directory):
            #     os.makedirs(directory)
            # plt.savefig(Args.HOME_DIR + '/Longitudinal_Classifier' + '/_dx_' + str(i) + '/_' + str(r) + '_' + str(j) + '_slope.png')
            plt.show()

        # Plot histogram of region r
        # r = 20
        # x_m = cc.m[:,r]
        # x_mp = cc.m_p[:,r]
        # x_c = cc.c[:,r]
        # plt.subplot(1, 3, i + 1)
        # plt.hist(x_m, density=True)
        # plt.hist(x_mp, density=True)
        # plt.hist(x_c, density=True)
        # plt.ylim([0, 15])
        # plt.title(str(i))


    # plt.show()
    # cc.get_gradient(cort_list)
    # cc.plot_fit(cort_list, m, c)
