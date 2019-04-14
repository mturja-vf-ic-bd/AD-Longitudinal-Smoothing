import matplotlib
import numpy as np
from matplotlib.patches import Circle, Wedge, Polygon
from matplotlib.collections import PatchCollection
import matplotlib.pyplot as plt
from utils.readFile import readSubjectFiles
import bct
from utils.helper import get_lobe_idx, get_sorted_node_count, get_top_links, get_lobe_order, get_subject_names, forward_diff
from utils.sortDetriuxNodes import sort_matrix
from utils.CurvedText import CurvedText

def get_box_circle_patches(center, rad, arc_ratio, c_arc, width, window=(0, 360)):
    patches = []
    start, end = window[0], window[1]
    r = end - start
    arc_l = arc_ratio * r / c_arc
    gap = (1 - arc_ratio) * r / c_arc

    for i in range(0, c_arc):
        patches += [
            Wedge(center, rad, start + i * (gap + arc_l), start + i * (gap + arc_l) + arc_l,
                  width=width)  # Ring sector
        ]

    return patches

def add_text(ax, patches, center):
    lobe_order = get_lobe_order(True)
    plt.rc('font', family='Times New Roman')
    for i, p in enumerate(patches):
        r, theta1, theta2 = p.r, p.theta1 * np.pi/180, p.theta2 * np.pi/180
        x, y = r * 1.05 * np.cos((theta1 + theta2)/2) + center[0], \
               r * 1.05 * np.sin((theta1 + theta2)/2) + center[1]

        ax.text(x, y, lobe_order[i], rotation=(p.theta1 + p.theta2)/2 - 90, horizontalalignment='center',
                verticalalignment='center', fontsize=25)

def plot_ring(color_list_face, color_list_edge, lobes):
    fig, ax = plt.subplots(figsize=(10, 10))
    patches = []

    # Some limiting conditions on Wedge
    center = (.5, .5)
    rad = 0.5
    colors_face = []
    colors_edge = []
    lobes = np.array(lobes)
    lobe_gap = 2
    lobes_ratio = lobes / sum(lobes)
    text_coord_patch = []
    for i, color in enumerate(color_list_face):
        start = 0
        total = 360
        new_patches = []
        for j, lobe in enumerate(lobes):
            if j >= len(lobes) / 2 and start < 180:
                start = 180
            end = start + total * lobes_ratio[j] - lobe_gap
            lobe_patch = get_box_circle_patches(center, rad, 0.8, lobe, 0.01,
                                                window=(start, end))
            new_patches += lobe_patch
            if i == 0:
                text_coord_patch.append(lobe_patch[int(len(lobe_patch)/2) - 1])
            start = end + lobe_gap

        patches += new_patches
        rad = rad * 0.97
        colors_face += list(color)
        colors_edge += list(color_list_edge[i])

    p = PatchCollection(patches, facecolors=colors_face, edgecolors=colors_edge, alpha=1)
    ax.add_collection(p)
    add_text(ax, text_coord_patch, center)
    ax.set_axis_off()

    return new_patches

def get_outlier_nodes(data, feat="deg", threshold=1e-3):
    T = len(data)
    n = len(data[0])
    feature = np.empty((T, n))
    for t in range(0, T):
        if feat == "deg":
            feature[t] = data[t].sum(axis=0)
        elif feat == "cent":
            feature[t] = bct.betweenness_wei(data[t])
        elif feat == "cc":
            feature[t] = bct.clustering_coef_wu(data[t])
        elif feat == "assort":
            feature[t] = bct.assortativity_wei(data[t])

    diff = forward_diff(forward_diff(feature))
    ind_vec = diff > threshold
    print(ind_vec.sum(axis=None))
    return ind_vec

def get_ring_colors(data):
    outlier = get_outlier_nodes(data)
    face_colors = []
    edge_colors = []

    for i in range(0, len(outlier)):
        face_color = []
        edge_color = []
        for j in range(0, len(outlier[i])):
            if outlier[i][j]:
                face_color.append('#ff0000')
                edge_color.append('#ff0000')
            else:
                face_color.append('#ffffff')
                edge_color.append('#003366')
        face_colors.append(face_color)
        edge_colors.append(edge_color)
    return face_colors, edge_colors

def get_edges(coord, links):
    edges = []
    for link in links:
        i, j, w = link
        edges.append((coord[i], coord[j], w))

    return edges

def plot_edges(edges):
    for edge in edges:
        x1, y1 = edge[0]
        x2, y2 = edge[1]
        w = edge[2]
        plt.plot([x1, x2], [y1, y2], marker='.', ls='-',
                 color='#804000', linewidth=w*500, alpha=0.5)

def plot_circle(data, edges, save=True, fname='circle_plot'):
    color_list_face_rw, color_list_edge_rw = get_ring_colors(data)
    lobes_count = get_sorted_node_count()
    inner_ring = plot_ring(color_list_face_rw, color_list_edge_rw, lobes_count)
    center = (.5, .5)
    coord = []
    for patch in inner_ring:
        theta = (patch.theta2 + patch.theta1) * np.pi / 360
        r = patch.r * 0.95
        x, y = r * np.cos(theta) + center[0], r * np.sin(theta) + center[1]
        coord.append((x, y))

    edges = get_edges(coord, edges)
    plot_edges(edges)
    if save:
        fig = plt.gcf()
        fig.tight_layout()
        fig.savefig(fname + '.png')
    else:
        plt.show()

def main(sub="027_S_5110"):
    rw, sm = readSubjectFiles(sub, "whole", sort=False)

    for t in range(0, len(rw)):
        rw[t], order = sort_matrix(rw[t], False)
        sm[t], order = sort_matrix(sm[t], False)

    # Plot Raw
    rw_links = get_top_links(rw[1], count=500, offset=0, weight=True)
    plot_circle(rw, rw_links, True, 'raw_cplot')

    # Plot Intrinsic
    sm_links = []
    for rl in rw_links:
        i, j, w = rl
        sm_links.append((i, j, sm[1][i, j]))

    plot_circle(sm, sm_links, True, 'sm_cplot')


if __name__ == '__main__':
    subname = get_subject_names(5)
    subname =["094_S_4162"]
    for sub in subname:
        print(sub)
        main(sub)
