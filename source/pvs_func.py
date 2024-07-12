import numpy as np
import time

EPSILON = 0.1


def make_ssect_graph(all_portals):
    ssect_graph = {}
    for n in all_portals:
        if n[0] not in ssect_graph:
            ssect_graph[n[0]] = []
        ssect_graph[n[0]].append((n[1], np.array(n[2], dtype='float'), np.array(n[3], dtype='float')))
        if n[1] not in ssect_graph:
            ssect_graph[n[1]] = []
        ssect_graph[n[1]].append((n[0], np.array(n[3], dtype='float'), np.array(n[2], dtype='float')))
    #
    portal_cantsee = {}
    ####skeys = sorted(ssect_graph.keys())
    ####for ssi in skeys:
    ####    for portal_dat in ssect_graph[ssi]:
    ####        v1 = portal_dat[2] - portal_dat[1]
    ####        plane = np.array([-v1[1], v1[0]], dtype='float')
    ####        mid = (portal_dat[1] + portal_dat[2])/2.
    ####        for ssj in skeys:
    ####            if ssi == ssj:
    ####                continue
    ####            for portal_dat2 in ssect_graph[ssj]:
    ####                p1 = np.dot(plane, portal_dat2[1] - mid)
    ####                if p1 > EPSILON:
    ####                    continue
    ####                p2 = np.dot(plane, portal_dat2[2] - mid)
    ####                if p2 > EPSILON:
    ####                    continue
    ####                portal_cantsee[(ssi, portal_dat[0], ssj, portal_dat2[0])] = True
    return (ssect_graph, portal_cantsee)


def precompute_portal_visibility(ssect_graph, my_inds, results_dict, print_progress=False):
    skeys = sorted(ssect_graph.keys())
    for ssi in my_inds:
        if ssi not in ssect_graph:
            continue
        tt = time.perf_counter()
        for portal_dat in ssect_graph[ssi]:
            v1 = portal_dat[2] - portal_dat[1]
            plane = np.array([-v1[1], v1[0]], dtype='float')
            mid = (portal_dat[1] + portal_dat[2])/2.
            for ssj in skeys:
                if ssi == ssj:
                    continue
                for portal_dat2 in ssect_graph[ssj]:
                    p1 = np.dot(plane, portal_dat2[1] - mid)
                    if p1 > EPSILON:
                        continue
                    p2 = np.dot(plane, portal_dat2[2] - mid)
                    if p2 > EPSILON:
                        continue
                    results_dict[(ssi, portal_dat[0], ssj, portal_dat2[0])] = True
        if print_progress:
            print(f'subsector {ssi}: {int(time.perf_counter() - tt)} sec')


def clip_target(tar, plane, plane_dist):
    dists = [np.dot(tar[0], plane) - plane_dist, np.dot(tar[1], plane) - plane_dist]
    if dists[0] < EPSILON and dists[1] < EPSILON:
        return None
    if dists[0] >= -EPSILON and dists[1] >= -EPSILON:
        return tar
    #
    p1 = tar[0]
    p2 = tar[1]
    dot = dists[0] / (dists[0] - dists[1])
    #
    mid = np.array([p1[0] + dot * (p2[0] - p1[0]), p1[1] + dot * (p2[1] - p1[1])], dtype='float')
    if dists[0] < -EPSILON:
        out_tar = [mid, tar[1]]
    else:
        out_tar = [tar[0], mid]
    return out_tar


def clip_to_separators(p_source, p_pass, p_target):
    out_target = [p_target[0], p_target[1]]
    for i in [0,1]:
        for j in [0,1]:
            v2 = p_pass[j] - p_source[i]
            plane = np.array([-v2[1], v2[0]], dtype='float')
            length = plane[0] * plane[0] + plane[1] * plane[1]
            if length < EPSILON: # invalid plane
                continue
            length = 1 / np.sqrt(length)
            plane *= length
            plane_dist = np.dot(p_pass[j], plane)
            #
            d = np.dot(p_source[i ^ 1], plane) - plane_dist
            if d > EPSILON:
                plane = -plane
                plane_dist = -plane_dist
            elif d >= -EPSILON: # planar with source portal
                continue
            #
            d = np.dot(p_pass[j ^ 1], plane) - plane_dist
            if d <= EPSILON: # planar with plane or points on negative side
                continue
            #
            out_target = clip_target(out_target, plane, plane_dist)
            if out_target is None:
                return None
    return out_target


def PVS_DFS(ssect_graph, starting_node, portal_cantsee={}):
    visited = {}
    leaves = {}
    stack = [(starting_node, [], None)]
    while stack:
        (node, path, previous_target) = stack.pop()
        #print(node, len(path), [n[0] for n in path], previous_target)
        new_target = None
        if len(path) >= 3:
            portal_source = [path[0][1], path[0][2]]
            if previous_target is None:
                portal_pass = [path[-2][1], path[-2][2]]
            else:
                portal_pass = previous_target
            portal_target = [path[-1][1], path[-1][2]]
            new_target = clip_to_separators(portal_source, portal_pass, portal_target)
            if new_target is None:
                leaves[node] = True
                continue
        visited[node] = True
        if node in ssect_graph: # graph can be missing the node if it has no neighbors
            path_nodes = {n[0]:True for n in path}
            for neighbor in [n for n in ssect_graph[node] if n[0] not in path_nodes]:
                all_cansee = True
                if len(path) >= 3:
                    for i in range(len(path)-3):
                        if (path[i][0], path[i+1][0], path[-1][0], neighbor[0]) in portal_cantsee:
                            all_cansee = False
                            break
                if all_cansee:
                    stack.append((neighbor[0], path+[neighbor], new_target))
    return sorted(visited.keys())


def PVS_DFS_parallel(ssect_graph, list_of_starting_nodes, results_dict, portal_cantsee={}, print_progress=False):
    for starting_node in list_of_starting_nodes:
        tt = time.perf_counter()
        results_dict[starting_node] = PVS_DFS(ssect_graph, starting_node, portal_cantsee=portal_cantsee)
        if print_progress:
            print(f'subsector {starting_node}: {int(time.perf_counter() - tt)} sec')
