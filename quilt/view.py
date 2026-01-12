import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import networkx as nx
import zipfile
import mne
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from matplotlib.colors import Normalize
from quilt.eeg import EEGholder

import pyvista as pv

class ParcelledCortexPlot:
    '''Plotter for a scalar field on the surface'''

    def __init__(self, surface_zipfile, connectivity_zipfile, vertex_mapping, 
                 screenshot_dpi=300, width_inches=5, height_inches=4):
        
        # Loads vetrices and triangles from the goddam zip file
        with zipfile.ZipFile(surface_zipfile, 'r') as zip_ref:
            vertices = np.loadtxt(zip_ref.open('vertices.txt'))
            triangles = np.loadtxt(zip_ref.open('triangles.txt')).astype(int)
        
        # Loads the oscillator names
        with zipfile.ZipFile(connectivity_zipfile, 'r') as zip_ref:
            self.nodenames = np.loadtxt(zip_ref.open('centres.txt'), dtype=str)
            self.nodenames = self.nodenames[:,0]

        # Suffered region mapping goes here
        self.mapping = np.loadtxt(vertex_mapping).astype(int)
        self.node_ids = np.sort(np.unique(self.mapping))

        # Match node-node_id
        self.node_node_id = {node:id_ for node, id_ in zip(self.nodenames, self.node_ids)}

        # Compute dict of mapping
        self.node_vertex_map = {}
        for node_id in self.node_ids:
            is_in_node = self.mapping == node_id
            self.node_vertex_map[node_id] = np.where(is_in_node)

        # Makes the pyvista mesh
        self.mesh = pv.PolyData()
        self.mesh.points = vertices
        self.mesh.faces = np.hstack([[3, *tr] for tr in triangles]).astype(int)

        # Default params
        self.screenshot_window_size = [int(screenshot_dpi * width_inches), int(screenshot_dpi * height_inches)]
        self.viz_window_size = [600, 600]
        self.mesh_viz_kwargs = dict( 
            specular=0.5,         
            specular_power=10,     
            diffuse=0.8,           
            ambient=0.2,      
            opacity=1,    
            show_edges=False,
            interpolate_before_map=False,
            smooth_shading=True,
        )

    def screenshot(self,name='screenshot.png'):
        self.plotter.camera.zoom(1.5)
        self.plotter.screenshot(name, window_size=self.screenshot_window_size)

    
    def set_scalar(self, scalar_field_dict, name='scalar'):
        if len(scalar_field_dict) != len(self.node_ids):
            raise ValueError("Scalar field has different size than nodes")

        # Assing vertex scalar based on mapping
        vertex_values = np.zeros(len(self.mesh.points))
        for node in scalar_field_dict:
            vertex_values[self.node_vertex_map[self.node_node_id[node]]] = scalar_field_dict[node]

        self.mesh.point_data[name] = vertex_values
    
    def plot(self, scalar_name='scalar', clim=None ,cmap=plt.cm.RdPu_r, 
             scalar_bar_title=None, scalar_bar_fontsize=9, scalar_bar=True):
        self.plotter = pv.Plotter(window_size=self.viz_window_size)
        self.plotter.enable_parallel_projection()
        self.plotter.set_background("white")

        actor = self.plotter.add_mesh(
                self.mesh,
                scalars=scalar_name,  
                cmap=cmap,
                show_scalar_bar=scalar_bar,
                scalar_bar_args=dict(font_family='times', 
                                     title=scalar_bar_title,
                                     title_font_size=scalar_bar_fontsize, 
                                     label_font_size=int(scalar_bar_fontsize/1.5)),
                clim=clim,
                **self.mesh_viz_kwargs
                )
    
    def show(self):
        self.plotter.show()

    
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
