# Network analysis
import networkx as nx
import bct
# Import external packages
import numpy as np

# Import matplotlib for visualization
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_agg import FigureCanvasAgg


def get_graph(dset, tmpobj, louvain=False):
    G = nx.Graph()
    G.add_nodes_from(zip(*tmpobj.label.values())[0][1:])
    for roi in dset.columns:
        for i, value in enumerate(dset[roi]):
            if not np.isnan(value):
                if value:
                    G.add_edge(roi, dset.index[i], weight=value)
    edges, weights = zip(*nx.get_edge_attributes(G, 'weight').items())

    if louvain:
        modules = dict(zip(dset.columns, bct.modularity_louvain_und(bct.binarize(dset.values, copy=True))[0]))
    else:
        modules = dict(zip(dset.columns, bct.modularity_und(dset.values)[0]))
    comm_idx = dict()
    for node, comm in modules.items():
        if comm not in comm_idx.keys():
            comm_idx[comm] = [node]
        else:
            comm_idx[comm].append(node)
    colors_set = colors.XKCD_COLORS.keys()[::-1]
    node_color_map = []
    for node in G.nodes():
        node_color_map.append(colors_set[modules[node]])
    return G, edges, weights, modules, node_color_map, comm_idx, colors_set


def plot_brain(G, node_color_mapped, comm_idx, weights, tmpobj, colors_set, alpha=0.4,
               vmin=-0.8, vmax=0.8, node_size=50):
    resol = np.array(tmpobj.image.dataobj.shape)
    edge_cmap = plt.cm.coolwarm
    coronal = dict()
    axial = dict()
    sagital = dict()
    for key, value in tmpobj.atlas.coordinates.items():
        if key == 'Clear Label':
            pass
        else:
            coronal[key] = value[:2][::-1]
            axial[key] = value[[0, 2]]
            sagital[key] = value[[1, 2]]
    figsize = [(resol[0] + resol[1]) / 20,
               (resol[1] + resol[2]) / 20]

    fig = plt.Figure(figsize=figsize, dpi=300)
    canvas = FigureCanvasAgg(fig)
    gs1 = gridspec.GridSpec(2, 2,
                            height_ratios=[resol[2], resol[0]],
                            width_ratios=[resol[1], resol[0]])
                            # width_ratios=[resol[0], resol[1]],
                            # height_ratios=[resol[1], resol[2]])
    gs2 = gridspec.GridSpecFromSubplotSpec(4, 3, subplot_spec=gs1[1, 1],
                                           width_ratios=[0.4, 1.5, 0.4],
                                           height_ratios=[0.2, 0.1, 0.3, 1.5])
    gs1.update(wspace=0.0, hspace=0.0, bottom=0.05, top=0.95, left=0.05, right=0.9)
    cbaxes = fig.add_subplot(gs2[1, 1])
    lgaxes = fig.add_subplot(gs2[3, 1])

    cor_ax = fig.add_subplot(gs1[1, 0])
    axl_ax = fig.add_subplot(gs1[0, 1])
    sag_ax = fig.add_subplot(gs1[0, 0])
    cor_ax.set_axis_off()
    cor_ax.set_facecolor('white')
    cor_ax.set_ylim([0, resol[0]])
    cor_ax.set_xlim([resol[1], 0])
    axl_ax.set_axis_off()
    axl_ax.set_facecolor('white')
    axl_ax.set_xlim([0, resol[0]])
    axl_ax.set_ylim([0, resol[2]])
    sag_ax.set_axis_off()
    sag_ax.set_facecolor('white')
    sag_ax.set_ylim([0, resol[2]])
    sag_ax.set_xlim([resol[1], 0])

    cor_ax.imshow(tmpobj.mask._image.dataobj.sum(axis=2),
                  cmap='Greys', alpha=alpha, interpolation="bicubic")
    axl_ax.imshow(tmpobj.mask._image.dataobj.sum(axis=1).T,
                  cmap='Greys', alpha=alpha, origin='lower', interpolation="bicubic")
    sag_ax.imshow(tmpobj.mask._image.dataobj.sum(axis=0).T,
                  cmap='Greys', alpha=alpha, interpolation="bicubic")

    nx.draw_networkx_nodes(G, nodelist=G.nodes(), pos=coronal,
                           with_labels=False, node_size=node_size,
                           node_color=node_color_mapped, ax=cor_ax)
    edges = nx.draw_networkx_edges(G, pos=coronal, edge_color=weights, edge_cmap=edge_cmap,
                                   edge_vmin=vmin, edge_vmax=vmax,
                                   width=(np.array(weights) + 0.7) * 1.5, ax=cor_ax)
    nx.draw_networkx_nodes(G, nodelist=G.nodes(), pos=axial,
                           with_labels=False, node_size=node_size,
                           node_color=node_color_mapped, ax=axl_ax)
    nx.draw_networkx_edges(G, pos=axial, edge_color=weights, edge_cmap=edge_cmap,
                           edge_vmin=vmin, edge_vmax=vmax,
                           width=(np.array(weights) + 0.7) * 1.5, ax=axl_ax)
    nx.draw_networkx_nodes(G, nodelist=G.nodes(), pos=sagital,
                           with_labels=False, node_size=node_size,
                           node_color=node_color_mapped, ax=sag_ax)
    nx.draw_networkx_edges(G, pos=sagital, edge_color=weights, edge_cmap=edge_cmap,
                           edge_vmin=vmin, edge_vmax=vmax,
                           width=(np.array(weights) + 0.7) * 1.5, ax=sag_ax)

    fig.colorbar(edges, cax=cbaxes, orientation='horizontal', ticks=[vmin, 0, vmax])
    fig.set_facecolor('white')

    # colors_set = colors.XKCD_COLORS.keys()[::-1]

    n_keys = len(comm_idx.keys())
    lgaxes.set_xlim(0, 3.5)
    lgaxes.set_ylim(np.around(n_keys / 4.)*1.7, 0)
    lgaxes.set_axis_off()

    n_half = np.around(n_keys / 2.)
    for i in range(0, n_keys):
        i += 1
        if i <= np.around(n_keys / 2.)+1:
            lgaxes.add_patch(patches.Rectangle((0.2, i / 2.), 0.3, 0.2,
                                               facecolor=colors_set[i]))
            lgaxes.text(0.7, i / 2. + 0.2 , str(i), fontsize=10)
        else:
            lgaxes.add_patch(patches.Rectangle((2.2, i / 2. - (n_half / 2.) - 0.5), 0.3, 0.2,
                                               facecolor=colors_set[i]))
            lgaxes.text(2.7, i / 2. - 0.3 - (n_half / 2.), str(i), fontsize=10)
    return fig