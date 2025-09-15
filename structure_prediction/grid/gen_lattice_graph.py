import pickle
import networkx as nx
import networkit as nk
import random
random.seed(42)



list_graph = []

for i in range(5):
    m, n  = random.randint(40, 70), random.randint(40, 70)
    graph = nx.hexagonal_lattice_graph(m, n)
    graph = nk.nxadapter.nx2nk(graph)
    print(graph)
    list_graph.append(graph)

for i in range(6):
    m, n  = random.randint(50, 80), random.randint(50, 80)
    graph = nx.triangular_lattice_graph(m, n)
    graph = nk.nxadapter.nx2nk(graph)
    print(graph)
    list_graph.append(graph)


for i in range(10,14):
    graph = nx.hypercube_graph(i)
    graph = nk.nxadapter.nx2nk(graph)
    print(graph)
    list_graph.append(graph)

print("Number of graphs:", len(list_graph))
with open("list_graph.pickle", 'wb') as f:
    pickle.dump(list_graph, f)
