import networkit as nk
from collections import Counter, defaultdict
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import time
from utils import random_walk, get_adjacency_matrix
import random
import tiktoken
import pickle
import json
random.seed(42)

def get_prompt(graph, list_start_nodes, walk_length):
    nodelist = sorted(list(graph.iterNodes()))
    adj_matrix = get_adjacency_matrix(graph, nodelist)

    initial_prompt = (f"You are a graph theory expert. Following are statistics of simple random walks performed on a large graph. The walks may not cover the whole graph.\n"
        f"For each walk, walk statistics are provided as a dictionary: {{<node name>:(<number of times node appears in the walk>, <degree of the node>), }}.\n"
        f"Use this information to predict if the graph is one of the following types: Barabasi-Albert (scale-free graph), Erdos-Renyi (random graph), LFR (Lancichinetti–Fortunato–Radicchi benchmark graph), or a grid graph (regular lattice). \n"
        f"Provide your answer in the following format, Graph Type: <type of graph> (choose one from: BA, ER, LFR, Grid).\n")


    for idx, start_node in enumerate(list_start_nodes):
        seed = idx
        list_walk = random_walk(adj_matrix, start_node, walk_length=walk_length, seed=seed)
        unique_walk_nodes = list(set(list_walk))
        degree_walk_nodes = [graph.degree(nn) for nn in unique_walk_nodes]
        rev_sort_idx = np.argsort(degree_walk_nodes)[::-1]

        count_walk = Counter(list_walk)
        #nodes are sorted by degree from highest to lowest
        list_walk_info = {unique_walk_nodes[nn]:(count_walk[unique_walk_nodes[nn]], degree_walk_nodes[nn]) for nn in rev_sort_idx}
        list_walk_info = str(list_walk_info).replace(" ", "")
        list_walk_info = list_walk_info.replace("),", "), ")

        initial_prompt += f"Walk {idx+1}: {str(list_walk_info)}\n"
        initial_prompt += f"Length of walk: {len(list_walk)}. Number of unique nodes in the walk: {len(unique_walk_nodes)}\n"

    encoding = tiktoken.get_encoding("o200k_base")
    num_tokens = len(encoding.encode(initial_prompt))
    print(f"Number of tokens in the prompt: {num_tokens}")

    return initial_prompt, num_tokens


with open('list_graph.pickle', 'rb') as f:
    list_graph = pickle.load(f)


dict_prompts = defaultdict(dict)

for ii, graph_nk in enumerate(list_graph):
    print(f"==== Graph: {ii} ====")


    start_nodes = random.sample(list(graph_nk.iterNodes()), 5)

    walk_length = int(graph_nk.numberOfNodes()*0.1)

    initial_prompt, num_tokens = get_prompt(graph_nk, start_nodes, walk_length)
    dict_prompts[ii]['graph_type'] = "BA"
    dict_prompts[ii]['prompt'] = initial_prompt
    dict_prompts[ii]['graph_size'] = graph_nk.numberOfNodes()
    dict_prompts[ii]['num_tokens'] = num_tokens


# Save the prompts to a file
with open("ba_prompts.json", "w") as f:
    json.dump(dict_prompts, f, indent=4)
