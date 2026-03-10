"""
subdivideTruss — Split each truss member into sub-elements.

Port of subdivideTruss.m from ShepherdLab LW FEA-OPT.
"""

import numpy as np


def subdivide_truss(nodes, connectivity, num_sub):
    """
    Split each truss member into num_sub sub-elements.

    Parameters
    ----------
    nodes : (N, 2) array
    connectivity : (M, 2) array, 0-based
    num_sub : int >= 1

    Returns
    -------
    new_nodes : (N2, 2) array
    new_conn : (M2, 2) array, 0-based
    member_map : (M2,) array — original member index for each sub-element
    """
    if num_sub < 1:
        raise ValueError("num_sub must be >= 1.")

    nodes = np.asarray(nodes, dtype=float)
    connectivity = np.asarray(connectivity, dtype=int)

    num_orig_nodes = nodes.shape[0]
    num_orig_members = connectivity.shape[0]

    if num_sub == 1:
        return nodes.copy(), connectivity.copy(), np.arange(num_orig_members)

    num_new_nodes = num_orig_nodes + num_orig_members * (num_sub - 1)
    num_new_elems = num_orig_members * num_sub

    new_nodes = np.zeros((num_new_nodes, 2))
    new_conn = np.zeros((num_new_elems, 2), dtype=int)
    member_map = np.zeros(num_new_elems, dtype=int)

    new_nodes[:num_orig_nodes] = nodes

    next_node = num_orig_nodes
    next_elem = 0

    for m in range(num_orig_members):
        n1, n2 = connectivity[m]
        p1, p2 = nodes[n1], nodes[n2]

        fracs = np.arange(1, num_sub) / num_sub

        int_ids = np.zeros(num_sub - 1, dtype=int)
        for k in range(num_sub - 1):
            new_nodes[next_node] = p1 + fracs[k] * (p2 - p1)
            int_ids[k] = next_node
            next_node += 1

        chain = np.concatenate([[n1], int_ids, [n2]])

        for k in range(num_sub):
            new_conn[next_elem] = [chain[k], chain[k + 1]]
            member_map[next_elem] = m
            next_elem += 1

    return new_nodes, new_conn, member_map
