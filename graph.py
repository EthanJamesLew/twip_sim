from networkx import nx
import matplotlib.pyplot as plt

from scipy.integrate import odeint
import numpy as np
from numpy import sin

from hyper_tree import HyperRectTree, draw_rect_tree

def duffing_system(pt, tspan):
    force = np.zeros((2,))
    force[0] = pt[1]
    force[1] =  - sin(pt[0]) - 0.3*pt[1]
    return force

def solve_system(dt, x, f, ct=0):
    tspan = (ct, ct + dt)
    sol =  odeint(f, x, tspan)
    return sol[-1, :]

def get_neighbor(covering, pt, dt, f, break_count=100):
    curr_cell = covering.get_cell(pt)
    next_cell = curr_cell
    new_pt = pt
    itr =0
    while(curr_cell == next_cell and itr <= break_count):
        new_pt = solve_system(dt, new_pt, f)
        next_cell = covering.get_cell(new_pt)
        itr += 1
    return next_cell

def uniform_subdivide(covering, depth):
    if covering.depth < depth:
        covering.subdivide()
        for c in covering.rects:
            uniform_subdivide(c, depth)

def construct_graph(covering, f, dt=1/60, n_runs = 3):
    # Get the middle points for all the cells
    mpts =  covering.get_all()
    n = covering.n
    mpts = np.reshape(np.array(mpts), (len(mpts)//n, n))

    # Create empty graph object
    sym_g = nx.DiGraph()

    # Add unstable node
    sym_g.add_node(-1, attr_dict=None)

    # Create a structure to reverse map attr to nodes
    attr_d = {}
    attr_d[None] = -1

    # Add nodes
    for i in range(0, np.shape(mpts)[0]):
        cell  = covering.get_cell(mpts[i, :])
        attr_d[cell] = i
        sym_g.add_node(i)
        nx.set_node_attributes(sym_g, {i : {'cell':cell}})
        #nx.set_node_attributes(sym_g, i, cell)

    # Add neighbor
    idx = 0
    for i in mpts:
        cell = covering.get_cell(i)
        pts = cell.get_even_per_cell(n_runs)
        for pt in pts:
            neigh = get_neighbor(covering, pt, dt, f)
            sym_g.add_edge(idx, attr_d[neigh])
        idx += 1

    return sym_g, attr_d
    

def find_recurrent_path(G):
    pass
    
if __name__ == "__main__":
    covering = HyperRectTree((-7, -7, 7, 7))
    uniform_subdivide(covering, 3)

    graph, attr_d = construct_graph(covering, duffing_system)

    for i in range(0, 3):
        # find simple cycles
        cycles = nx.simple_cycles(graph)

        # extract recurrent vertices from cycles
        recurrent_vertices = set()
        for cycle in cycles:
            recurrent_vertices |= set(cycle)

        presd = []
        new_nodes = []
        for v in recurrent_vertices:
            # get predessors of subdivided nodes
            pres = graph.predecessors(v)
            presd = presd + list(pres)

        # subdivide state space and modify its symbolic image
        for v in recurrent_vertices:
            # get dimensions of system
            n = covering.n

            # divide region
            cell = graph._node[v]['cell']
            cell.subdivide()

            

            # remove nodes 
            graph.remove_node(v)

            # register new nodes
            mpts = cell.get_all()
            mpts = np.reshape(np.array(mpts), (len(mpts)//n, n))
            M = graph.number_of_nodes()
            for i in range(0, len(mpts)):
                sub_cell = cell.get_cell(mpts[i, :])
                attr_d[sub_cell] = i+M+1
                graph.add_node(i+M+1)
                nx.set_node_attributes(graph, {i+M+1 : {'cell':sub_cell}})
                new_nodes.append(i+M+1)

        # check where predecessors and new nodes map to 
        idx = 0
        for p in presd+new_nodes:
            if p in recurrent_vertices:
                break
            cell = graph._node[p]['cell']
            pts = cell.get_even_per_cell(3)
            for pt in pts:
                neigh = get_neighbor(covering, pt, 1/60, duffing_system)
                successor = attr_d[neigh]
                #print(successor)
                graph.add_edge(idx, successor)
            idx += 1


    draw_rect_tree(covering)


    #nx.draw(graph, with_labels=True)
    plt.show()

    pt = [0.9,0.9]
    print(get_neighbor(covering, pt, 1/60, duffing_system))