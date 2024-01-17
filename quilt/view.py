import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import networkx as nx

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
        G.add_node(pop['name'])


    for proj in projections:
        efferent, afferent = proj['name'].split("->")
        efferent, afferent = efferent.strip(), afferent.strip()
        edge_dict = dict()
        if "weight_inh" in proj['features']:
            edge_dict = dict(   color="b", 
                                weight = proj['features']['weight_inh'],
                                lenght = proj['features']['delay'],
                                strongness = proj['features']['inh_fraction']*proj['features']['weight_inh']
                             )
        else:
            edge_dict = dict(   color="r", 
                                weight = proj['features']['weight_exc'],
                                lenght = proj['features']['delay'],
                                strongness = proj['features']['exc_fraction']*proj['features']['weight_exc']
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
