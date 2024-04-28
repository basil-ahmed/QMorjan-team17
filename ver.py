import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from gurobi_optimods.qubo import solve_qubo
from utils_dirac import wes_run_qubo

# Make a random graph
G = nx.erdos_renyi_graph(12, 0.3)

def build_vertex_cover_qubo(G, alpha, beta):
    n = len(G.nodes)
    def build_A_qubo(G):
        A = np.zeros((n, n))
    
        for (i, j) in G.edges:
            A[i, j] += 1/2
            A[j, i] += 1/2
            A[i, i] -= 1
            A[j, j] -= 1
        return A 

    def build_B_qubo(G):
        B = np.ones((n))
        B = np.diag(B)
        return B
    
    qubo = alpha * build_A_qubo(G) + beta * build_B_qubo(G)
    return qubo

qubo = build_vertex_cover_qubo(G, 1, 0.8)

def build_set_cover_qubo(G, alpha, beta, gamma):
    def build_subsets(G):
        subsets = []
        k = np.zeros(len(G.nodes), int)
        for i in range(len(G.nodes)):
            subsets.append([i])
            k[i] += 1
            for j in range(len(G.nodes)):
                if (i, j) in G.edges:
                    subsets[-1].append(j)
                    k[j] += 1
        return subsets, k

    subsets, k = build_subsets(G)
    n = len(G.nodes)
    length = n + sum(k)

    def build_A_qubo():
        position = n
        A = np.zeros((length, length))
        for i in range(n):
            size = k[i]
            X = np.ones((size, size)) - 2* np.diag(np.ones(size))
            # Append X to A in the position (position, position)
            A[position:position+size, position:position+size] = X
            position += size
        return A
    
    def build_B_qubo():
        B = np.zeros((length, length))
        position = n
        for i, subset in enumerate(subsets):
            array = np.zeros(length)
            for j in subset:
                array[j] = -1
            for j in range(k[i]):
                array[position + j] = j+1
            position += k[i]
            B += np.outer(array, array)
        return B  
        
    def build_C_qubo():
        C = np.zeros((length))
        C[:n] = 1
        C = np.diag(C)
        return C

    qubo = alpha * build_A_qubo() + beta * build_B_qubo() + gamma * build_C_qubo()
    return qubo

# qubo = build_set_cover_qubo(G, 1, 1, 0.8)
print(qubo)
# result = solve_qubo(qubo) 
# print(result)

def process_result(result, G, name='vertex_cover'):
    n = len(G.nodes)
    solution = result
    for i in range(n):
        if solution[i] == 1:
            print(i)
    # Color the nodes in the vertex cover
    colors = []
    for i in range(n):
        if solution[i] == 1:
            colors.append('red')
        else:
            colors.append('blue')
    # Erase the previous plot
    plt.clf()
    nx.draw(G, with_labels=True, node_color=colors)
    # Save the plot
    plt.savefig(f'{name}.png')

result_gurobi = solve_qubo(qubo).solution
process_result(result_gurobi, G, "first")

# Run with Dirac
# result_wes = wes_run_qubo(qubo, 1)
# process_result(np.array(result_wes[0]), G, "dirac")

