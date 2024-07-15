import numpy as np
import time

from collections import deque

EPSILON = 0.1


def precompute_portal_visibility(ssect_graph, portal_coords, ssi, print_progress=False):
    n_portals = portal_coords.shape[0]
    dict_out = {}
    tt = time.perf_counter()
    for (_, portal_i) in ssect_graph[ssi]:
        pi_p1 = portal_coords[portal_i,0:2]
        pi_p2 = portal_coords[portal_i,2:4]
        v1 = pi_p2 - pi_p1
        plane = np.array([-v1[1], v1[0]], dtype='float')
        length = plane[0] * plane[0] + plane[1] * plane[1]
        if length < EPSILON: # invalid plane
            continue
        length = 1 / np.sqrt(length)
        plane *= length
        mid = (pi_p1 + pi_p2)/2.0
        for ssj in range(len(ssect_graph)):
            if ssi == ssj:
                continue
            for (_, portal_j) in ssect_graph[ssj]:
                pj_p1 = portal_coords[portal_j,0:2]
                pj_p2 = portal_coords[portal_j,2:4]
                test_p1 = np.dot(plane, pj_p1 - mid)
                test_p2 = np.dot(plane, pj_p2 - mid)
                cantsee = False
                if test_p1 > EPSILON or test_p2 > EPSILON: # portal_j is on the right side of portal_i
                    v2 = pj_p2 - pj_p1
                    plane2 = np.array([-v2[1], v2[0]], dtype='float')
                    length2 = plane2[0] * plane2[0] + plane2[1] * plane2[1]
                    if length2 < EPSILON: # invalid plane
                        continue
                    length2 = 1 / np.sqrt(length2)
                    plane2 *= length2
                    test_v2 = abs(plane2[0] + plane[0]) + abs(plane2[1] + plane[1])
                    if test_v2 < EPSILON: # portal_j is facing opposite direction of portal_i
                        cantsee = True
                else:
                    cantsee = True
                if cantsee:
                    ind = (portal_i*n_portals + portal_j) // 8
                    bit = (portal_i*n_portals + portal_j) % 8
                    if ind not in dict_out:
                        dict_out[ind] = []
                    dict_out[ind].append(1 << bit)
    if print_progress:
        print(f'[precompute_portal_visibility] subsector {ssi}: {int(time.perf_counter() - tt)} sec')
    return dict_out


def count_neighbors(graph, starting_node, depth=3):
    queue = deque([(starting_node, [])])
    visited = {}
    while queue:
        (node, path) = queue.popleft()
        visited[node] = True
        if len(path) < depth:
            for neighbor in [n for n in graph[node] if n[0] not in visited]:
                queue.append((neighbor[0], path+[node]))
    return (len(visited), starting_node)


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


def PVS_DFS(ssect_graph, portal_coords, starting_node, portal_cantsee=None, results_thus_far=None, out_str_type=''):
    tt = time.perf_counter()
    n_portals = portal_coords.shape[0]
    visited = {}
    stack = [(starting_node, [], None)]
    while stack:
        (node, path, previous_target) = stack.pop()
        #print(starting_node, node, len(path), path, previous_target)
        new_target = None
        if len(path) >= 3:
            portal_source = [portal_coords[path[0][1],0:2], portal_coords[path[0][1],2:4]]
            if previous_target is None:
                portal_pass = [portal_coords[path[-2][1],0:2], portal_coords[path[-2][1],2:4]]
            else:
                portal_pass = previous_target
            portal_target = [portal_coords[path[-1][1],0:2], portal_coords[path[-1][1],2:4]]
            new_target = clip_to_separators(portal_source, portal_pass, portal_target)
            if new_target is None:
                continue
        visited[node] = True
        # if we've previously analyzed this subsector, we can ask whether or not it sees anything we haven't visited
        if results_thus_far is not None and results_thus_far[node]:
            nothing_new = True
            for visible_node in results_thus_far[node]:
                if visible_node not in visited:
                    nothing_new = False
                    break
            if nothing_new:
                continue
        #
        path_nodes = {n[0]:True for n in path}
        for neighbor in [n for n in ssect_graph[node] if n[0] not in path_nodes]:
            # can every portal in our path thus far potentially see the portal into this neighbor?
            all_cansee = True
            if portal_cantsee is not None:
                portal_j = neighbor[1]
                for (_, portal_i) in path:
                    ind = (portal_i*n_portals + portal_j) // 8
                    bit = (portal_i*n_portals + portal_j) % 8
                    if portal_cantsee[ind] & (1 << bit):
                        all_cansee = False
                        break
            #print('-', neighbor, all_cansee)
            if all_cansee:
                stack.append((neighbor[0], path+[neighbor], new_target))
    out_str = ''
    if out_str_type:
        out_str = f'[PVS_DFS] {out_str_type} {starting_node}: {int(time.perf_counter() - tt)} sec'
    return (starting_node, sorted(visited.keys()), out_str)
