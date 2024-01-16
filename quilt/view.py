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
        if "weight_inh" in proj['features']:
            color = "b"
            weight = proj['features']['delay']
        else:
            color = "r"
            weight = proj['features']['delay']
        G.add_edge(efferent,afferent,color=color,weight=weight)


    pos = nx.spring_layout(G)  # Posizionamento dei nodi con l'algoritmo Spring
    nx.draw(G, pos, 
            edge_color=nx.get_edge_attributes(G,'color').values(),
            with_labels=True, 
            font_weight='bold', 
            node_size=700, 
            node_color='skyblue', 
            font_size=8, 
            connectionstyle="arc3,rad=0.1",
            width=list(0.5*np.array(list(nx.get_edge_attributes(G, 'weight').values()))**2)
            )
