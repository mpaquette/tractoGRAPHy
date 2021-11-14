#!/usr/bin/env python3

import argparse

import numpy as np
import igraph as ig

from utils import load_graph



DESCRIPTION = """
Inspect graph.
"""

EPILOG = """
Michael Paquette, MPI CBS, 2021.
"""

class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawTextHelpFormatter):
    pass

def buildArgsParser():

    p = argparse.ArgumentParser(description=DESCRIPTION,
                                epilog=EPILOG,
                                formatter_class=CustomFormatter)

    p.add_argument('graph', type=str, default=[],
                   help='Path of the graph (pickled).')

    return p


def main():

    parser = buildArgsParser()
    args = parser.parse_args()

    graph_fname = args.graph

    g, vertex2vox, vox2vertex = load_graph(graph_fname)


    # number of nodes
    print('Graph has {:} nodes'.format(len(g.vs)))

    # number of edges
    print('Graph has {:} edges'.format(len(g.es)))

    # does the graph seem oriented?
    node1_name_len = len(g.vs['name'][1])
    if node1_name_len == 3:
        print('Graph seem naive type')
    elif node1_name_len ==4:
        print('Graph seem oriented type')

    # does the graph have a out-of-mask node
    try:
        out_of_mask_vertex = g.vs['name'].index('out-of-mask')
    except ValueError:
        out_of_mask_vertex = None

    if out_of_mask_vertex is not None:
        print('Graph has out-of-mask node')
    else:
        print('Graph doesnt have out-of-mask node')


if __name__ == "__main__":
    main()
