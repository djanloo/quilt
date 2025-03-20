import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import networkx as nx
import zipfile
import mne
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from matplotlib.colors import Normalize
from eeg import EEGholder

import pyvista as pv

def animate_spiking(spiking_network):

    fig, ax = plt.subplots()
    pop_vertices = []
    activities = []
    for pop in spiking_network.populations:
        pop_vertices.append(np.random.uniform(0,1, size=2))
        activities.append(np.array(spiking_network.populations[pop].get_data()["spikes"]))

    activities = np.array(activities)
    pop_vertices = np.array(pop_vertices)
    scat = ax.scatter(*(pop_vertices.T), c=activities[:, 0], s=800, vmin=0, vmax = 200)

    print(activities)

    def update(i):
        ax.set_title(f"t = {i}")
        scat.set_array(activities[:, i])

    anim = FuncAnimation(fig, update, frames=500, interval=10)
    return anim

def plot_graph(network):

    pops = network.features_dict['populations']
    projections = network.features_dict['projections']

    G = nx.DiGraph()

    for pop in pops:
        G.add_node(pop)


    for proj in projections:
        efferent, afferent = proj.split("->")
        efferent, afferent = efferent.strip(), afferent.strip()
        edge_dict = dict()

        features = projections[proj]
        if features['type'] == "inh":
            edge_dict = dict(   color="b", 
                                weight = features['weight'],
                                lenght = features['delay'],
                                strongness = features['connectivity']*features['weight']
                             )
        else:
            edge_dict = dict(   color="r", 
                                weight = features['weight'],
                                lenght = features['delay'],
                                strongness = features['connectivity']*features['weight']
                             )
        G.add_edge(efferent,afferent, **edge_dict)


    pos = nx.kamada_kawai_layout(G, weight='delay')

    linewidths = np.log(60*np.array(list(nx.get_edge_attributes(G, 'strongness').values())))
    linewidths = 0.1 + 3*(linewidths- np.min(linewidths))/(np.max(linewidths) - np.min(linewidths))
    nx.draw(G, pos, 
            edge_color=nx.get_edge_attributes(G,'color').values(),
            with_labels=True, 
            font_weight='bold', 
            node_size=700, 
            node_color='skyblue', 
            font_size=8, 
            connectionstyle="arc3,rad=0.1",
            width=linewidths
            )


def plot_neural_field(regional_attribute, surface_zipfile=None, region_mapping=None):

    regions = list(regional_attribute.keys())

    with zipfile.ZipFile(surface_zipfile, 'r') as zip_ref:
            vertices = np.loadtxt(zip_ref.open('vertices.txt'))
            triangles = np.loadtxt(zip_ref.open('triangles.txt')).astype(int)

    mapping = np.loadtxt(region_mapping).astype(int)

    mesh = pv.PolyData()
    mesh.points = vertices

    faces = np.hstack([[3, *tr] for tr in triangles]).astype(int)
    mesh.faces = faces

    colors = list(regional_attribute).values()

    color_faces = np.zeros(len(triangles))
    for i in range(len(triangles)):
        if mapping[triangles[i][0]] == mapping[triangles[i][1]] and mapping[triangles[i][1]] == mapping[triangles[i][2]]:
            try:
                color_faces[i] = colors[mapping[triangles[i][0]]]
            except IndexError as e:
                break
                print(triangles[i][0], end=" ", flush=True)

    plotter = pv.Plotter(notebook=True)
    plotter.add_mesh(mesh, scalars=color_faces, cmap="plasma", show_edges=False)

def plot_mutual_info(eeg : EEGholder, threshold, ax=None):
    """Plots the network induced mby the thresholded MI matrix """

    # If the MI is not computed yet
    if eeg.MI is None:
        print("Computing mutual info using bins")
        eeg.compute_mutual_info(bins=15)

    if ax is None:
        fig, ax = plt.subplots()

    G = nx.Graph()
    pos2d = eeg.pos_azimuthal
    for i in range(eeg.n_channels):
        G.add_node(eeg.channel_names[i], pos=pos2d[i])

    for i in range(eeg.n_channels):
        for j in range(i + 1, eeg.n_channels):
            if eeg.MI[i, j] > threshold:
                G.add_edge(eeg.channel_names[i], eeg.channel_names[j], weight=eeg.MI[i, j])

    pos_net = {eeg.channel_names[i]: (pos2d[i, 0], pos2d[i, 1]) for i in range(eeg.n_channels)}
    edges = G.edges(data=True)

    # Color is MI
    edge_colors = [data['weight'] for _, _, data in edges]
    # Line thickness is \propto MI
    edge_lws = [2*data['weight'] for _, _, data in edges]

    # Plots an empty field to show just the head
    mne.viz.plot_topomap( eeg.MI.diagonal()*0, pos2d, show=False, 
                            cmap="Greys", sphere=0.5,
                            extrapolate="head", axes=ax)

    nx.draw_networkx_nodes(G, pos_net, node_color="k", node_size=1, ax=ax)
    alpha = np.round(MinMaxScaler((0.01, 1)).fit_transform(np.array(edge_colors)[:,None]).squeeze(), 3)

    nx.draw_networkx_edges(G, pos_net, 
            alpha = alpha,
            edge_color=edge_colors, width=edge_lws,
            edge_cmap=plt.cm.rainbow,ax=ax)

    plt.colorbar(plt.cm.ScalarMappable(cmap=plt.cm.rainbow, norm=Normalize(vmin=np.min(edge_colors), vmax=np.max(edge_colors))), 
                ax=ax, label="Mutual Information")

def plot_entropy(eeg: EEGholder, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    
    if eeg.entropy is None:
        print("Computing entropy using bins")
        eeg.compute_entropy(bins=15)

    pos2d = eeg.pos_azimuthal
    mne.viz.plot_topomap( eeg.entropy, pos2d, show=False, 
                        cmap="rainbow", sphere=0.5,
                        extrapolate="head", axes=ax)
    
    plt.colorbar(plt.cm.ScalarMappable(cmap=plt.cm.rainbow, norm=Normalize(vmin=np.min(eeg.entropy), vmax=np.max(eeg.entropy))), 
                ax=ax, label="Entropy")
