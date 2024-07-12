#!/usr/bin/env python
import argparse
import copy
import multiprocessing
import numpy as np
import time

from source.pvs_func import precompute_portal_visibility, PVS_DFS_parallel
from source.wad_func import *


def main(raw_args=None):
    parser = argparse.ArgumentParser(description='pvs-reject', formatter_class=argparse.ArgumentDefaultsHelpFormatter,)
    parser.add_argument('-i', type=str, required=True,  metavar='input.wad',  help="* Input WAD")
    parser.add_argument('-m', type=str, required=True,  metavar='MAP01',      help="* Map name")
    parser.add_argument('-r', type=str, required=True,  metavar='REJECT.lmp', help="* Output reject table lmp")
    parser.add_argument('-p', type=int, required=False, metavar='4',          help="Number of processes to use", default=4)
    parser.add_argument('--plot-rej',   required=False, action='store_true',  help="Plot reject (for debugging)", default=False)
    parser.add_argument('--plot-ssect', required=False, action='store_true',  help="Plot subsectors (for debugging)", default=False)
    parser.add_argument('--progress',   required=False, action='store_true',  help="Print PVS progress to console", default=False)
    args = parser.parse_args()
    #
    IN_WAD = args.i
    WHICH_MAP = args.m
    OUT_REJECT = args.r
    NUM_PROCESSES = args.p
    PLOT_REJECT = args.plot_rej
    PLOT_SSECT = args.plot_ssect
    PRINT_PROGRESS = args.progress

    if PLOT_REJECT or PLOT_SSECT:
        import matplotlib.pyplot as mpl
        from matplotlib import collections as mc

    map_data = get_map_lmps(IN_WAD, WHICH_MAP)

    line_list = get_linedefs(map_data)
    side_list = get_sidedefs(map_data)
    sect_list = get_sectors(map_data)
    normal_verts = get_vertexes(map_data)
    gl_verts = get_gl_verts(map_data)
    ssect_list = get_gl_subsectors(map_data)
    segs_list = get_gl_segs_with_coordinates(map_data, normal_verts, gl_verts)
    #
    n_sectors = len(sect_list)
    n_subsectors = len(ssect_list)

    (portal_ssects, portal_coords, ssect_2_sect, segs_to_plot) = get_portal_segs(segs_list, ssect_list, line_list, side_list)
    n_portals = portal_ssects.shape[0]
    print(f'{n_sectors} sectors / {n_subsectors} subsectors / {n_portals} portals')

    ssect_graph = [[] for n in range(n_subsectors)]
    for i in range(portal_ssects.shape[0]):
        ssect_graph[portal_ssects[i,0]].append((portal_ssects[i,1], i))

    portal_cantsee = precompute_portal_visibility(ssect_graph, portal_ssects, portal_coords, PRINT_PROGRESS)

    tt = time.perf_counter()
    manager = multiprocessing.Manager()
    results_dict = manager.dict()
    processes = []
    for i in range(NUM_PROCESSES):
        my_inds = range(i, n_subsectors, NUM_PROCESSES)
        p = multiprocessing.Process(target=PVS_DFS_parallel, args=(ssect_graph, portal_coords, my_inds, results_dict, portal_cantsee, PRINT_PROGRESS))
        processes.append(p)
    for i in range(NUM_PROCESSES):
        processes[i].start()
    for i in range(NUM_PROCESSES):
        processes[i].join()
    print(f'PVSs finished: {int(time.perf_counter() - tt)} sec')

    reject_out = np.zeros((n_sectors, n_sectors), dtype='bool') + IS_INVISIBLE
    for i in range(n_subsectors):
        si = ssect_2_sect[i]
        for j in results_dict[i]:
            sj = ssect_2_sect[j]
            reject_out[si,sj] = IS_VISIBLE
            reject_out[sj,si] = IS_VISIBLE
    write_reject(reject_out, OUT_REJECT)

    #
    if PLOT_REJECT:
        fig = mpl.figure(0,figsize=(10,10))
        Z = reject_out
        X, Y = np.meshgrid(range(0,len(Z[0])+1), range(0,len(Z)+1))
        mpl.pcolormesh(X, Y, Z, cmap='binary', vmin=0, vmax=1)
        mpl.axis([0,len(Z[0]),0,len(Z)])
        mpl.gca().invert_yaxis()
        mpl.savefig(f'{OUT_REJECT}.png')
        mpl.close(fig)
    #
    if PLOT_SSECT:
        tt = time.perf_counter()
        for ssect_to_plot in range(n_subsectors):
            segs_to_plot_copy = copy.deepcopy(segs_to_plot)
            for ssi,ssect in enumerate(ssect_list):
                if ssi in results_dict[ssect_to_plot]:
                    my_segs = segs_list[ssect[1]:ssect[1]+ssect[0]]
                    for si,seg in enumerate(my_segs):
                        if ssi == ssect_to_plot:
                            segs_to_plot_copy.append([[seg[0], seg[1]], [0,0,0,1], 2.0])
                        else:
                            segs_to_plot_copy.append([[seg[0], seg[1]], [1,0,0,1], 1.0])
            #
            fig = mpl.figure(1, figsize=(10,10))
            lines  = [n[0] for n in segs_to_plot_copy]
            clines = [n[1] for n in segs_to_plot_copy]
            widths = [n[2] for n in segs_to_plot_copy]
            lc = mc.LineCollection(lines, colors=clines, linewidths=widths)
            mpl.gca().add_collection(lc)
            mpl.axis('scaled')
            mpl.savefig(f'{OUT_REJECT}.{ssect_to_plot}.png')
            mpl.close(fig)
        print(f'plotting finished: {int(time.perf_counter() - tt)} sec')


if __name__ == '__main__':
    main()
