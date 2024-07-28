import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import networkx as nx
import zipfile

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