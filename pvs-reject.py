#!/usr/bin/env python
import argparse
import copy
import numpy as np
import time

from concurrent.futures import as_completed, ProcessPoolExecutor

from source.pvs_func import count_neighbors, precompute_portal_visibility, PVS_DFS
from source.wad_func import *


def main(raw_args=None):
    parser = argparse.ArgumentParser(description='pvs-reject', formatter_class=argparse.ArgumentDefaultsHelpFormatter,)
    parser.add_argument('-i', type=str, required=True,  metavar='input.wad',  help="* Input WAD")
    parser.add_argument('-m', type=str, required=True,  metavar='MAP01',      help="* Map name")
    parser.add_argument('-r', type=str, required=True,  metavar='REJECT.lmp', help="* Output reject table lmp")
    parser.add_argument('-v', type=str, required=False, metavar='vis.npz',    help="Load precomputed visibility matrix", default='')
    parser.add_argument('-p', type=int, required=False, metavar='4',          help="Number of processes to use", default=4)
    parser.add_argument('--sector-dfs', required=False, action='store_true',  help="Use sector portals instead of subsectors", default=False)
    parser.add_argument('--save-vis',   required=False, action='store_true',  help="Save precomputed visibility matrix", default=False)
    parser.add_argument('--save-pvs',   required=False, action='store_true',  help="Save PVS results are they're computed", default=False)
    parser.add_argument('--plot-rej',   required=False, action='store_true',  help="Plot reject (for debugging)", default=False)
    parser.add_argument('--plot-ssect', required=False, action='store_true',  help="Plot subsectors (for debugging)", default=False)
    parser.add_argument('--progress',   required=False, action='store_true',  help="Print PVS progress to console", default=False)
    args = parser.parse_args()
    #
    IN_WAD = args.i
    WHICH_MAP = args.m
    OUT_REJECT = args.r
    LOAD_VISIBILITY = args.v
    NUM_PROCESSES = args.p
    SECTOR_MODE = args.sector_dfs
    SAVE_VISIBILITY = args.save_vis
    SAVE_PVS = args.save_pvs
    PLOT_REJECT = args.plot_rej
    PLOT_SSECT = args.plot_ssect
    PRINT_PROGRESS = args.progress

    if PLOT_REJECT or PLOT_SSECT:
        import matplotlib.pyplot as mpl
        from matplotlib import collections as mc

    map_data = get_map_lmps(IN_WAD, WHICH_MAP)
    if not map_data:
        print(f'Error: {WHICH_MAP} not found.')
        exit(1)

    line_list = get_linedefs(map_data)
    side_list = get_sidedefs(map_data)
    sect_list = get_sectors(map_data)
    normal_verts = get_vertexes(map_data)
    try:
        gl_verts = get_gl_verts(map_data)
        ssect_list = get_gl_subsectors(map_data)
        segs_list = get_gl_segs_with_coordinates(map_data, normal_verts, gl_verts)
    except KeyError:
        print(f'Error: {WHICH_MAP} does not have GL nodes.')
        exit(1)
    #
    (portal_ssects, portal_coords, ssect_2_sect, segs_to_plot) = get_portal_segs(segs_list, ssect_list, line_list, side_list)

    # cleanup
    n_sectors = len(sect_list)
    n_subsectors = len(ssect_list)
    n_portals = portal_ssects.shape[0]
    del map_data
    del line_list
    del side_list
    del sect_list
    del normal_verts
    del gl_verts
    print(f'{n_sectors} sectors / {n_subsectors} subsectors / {n_portals} portals')

    tt = time.perf_counter()
    if SECTOR_MODE:
        sect_graph = [[] for n in range(n_sectors)]
        for i in range(portal_ssects.shape[0]):
            my_sector = ssect_2_sect[portal_ssects[i,0]]
            partner_sector = ssect_2_sect[portal_ssects[i,1]]
            if my_sector != partner_sector:
                sect_graph[my_sector].append((partner_sector, i))
    #
    ssect_graph = [[] for n in range(n_subsectors)]
    for i in range(portal_ssects.shape[0]):
        ssect_graph[portal_ssects[i,0]].append((portal_ssects[i,1], i))
    sorted_ssect_nodes = sorted([count_neighbors(ssect_graph, n) for n in range(n_subsectors)])
    sorted_ssect_nodes = [n[1] for n in sorted_ssect_nodes]
    print(f'subsector graph finished: {int(time.perf_counter() - tt)} sec')

    if LOAD_VISIBILITY:
        tt = time.perf_counter()
        in_npz = np.load(LOAD_VISIBILITY)
        portal_cantsee = in_npz['portal_cantsee']
        print(f'precomputed visibility loaded from file: {int(time.perf_counter() - tt)} sec')
        print(f' - {portal_cantsee.shape[0]} bytes')
    else:
        tt = time.perf_counter()
        portal_cantsee = np.zeros(((n_portals*n_portals)//8 + 1), dtype='B')
        with ProcessPoolExecutor(max_workers=NUM_PROCESSES) as executor:
            futures = [executor.submit(precompute_portal_visibility, ssect_graph, portal_coords, n, PRINT_PROGRESS) for n in range(n_subsectors)]
            for future in as_completed(futures):
                ib_dict = future.result()
                for ind in ib_dict.keys():
                    for bit in ib_dict[ind]:
                        portal_cantsee[ind] |= bit
        if SAVE_VISIBILITY:
            np.savez_compressed(f'{OUT_REJECT}.npz', portal_cantsee=portal_cantsee)
        print(f'portal visibility precomputation finished: {int(time.perf_counter() - tt)} sec')
        print(f' - {portal_cantsee.shape[0]} bytes')

    if SAVE_PVS:
        f = open(f'{OUT_REJECT}.pvs', 'w')
        f.close()
    #
    # sector-based portal visibility (not as accurate, but possibly faster?)
    #
    if SECTOR_MODE:
        tt = time.perf_counter()
        pvs_result = [[] for _ in range(n_sectors)]
        with ProcessPoolExecutor(max_workers=NUM_PROCESSES) as executor:
            futures = [executor.submit(PVS_DFS, sect_graph, portal_coords, n, portal_cantsee, pvs_result, 'sector'*PRINT_PROGRESS) for n in range(n_sectors)]
            for future in as_completed(futures):
                (my_si, my_visited, progress_str) = future.result()
                pvs_result[my_si] = my_visited
        reject_out = np.zeros((n_sectors, n_sectors), dtype='bool') + IS_INVISIBLE
        for si in range(n_sectors):
            for sj in pvs_result[si]:
                reject_out[si,sj] = IS_VISIBLE
                reject_out[sj,si] = IS_VISIBLE
        print(f'PVSs finished: {int(time.perf_counter() - tt)} sec')
    #
    # subsector-based portal visibility
    #
    else:
        tt = time.perf_counter()
        ssect_processed_thus_far = 0
        pvs_result = [[] for _ in range(n_subsectors)]
        with ProcessPoolExecutor(max_workers=NUM_PROCESSES) as executor:
            futures = [executor.submit(PVS_DFS, ssect_graph, portal_coords, n, portal_cantsee, pvs_result, 'subsector'*PRINT_PROGRESS) for n in sorted_ssect_nodes]
            for future in as_completed(futures):
                (my_ssi, my_visited, progress_str) = future.result()
                pvs_result[my_ssi] = my_visited
                if SAVE_PVS:
                    with open(f'{OUT_REJECT}.pvs', 'a') as f:
                        f.write(str(my_ssi) + '\t' + ','.join([str(n) for n in my_visited]) + '\n')
                ssect_processed_thus_far += 1
                if PRINT_PROGRESS:
                    print(f'{progress_str} ({ssect_processed_thus_far}/{n_subsectors})')
        reject_out = np.zeros((n_sectors, n_sectors), dtype='bool') + IS_INVISIBLE
        for i in range(n_subsectors):
            si = ssect_2_sect[i]
            for j in pvs_result[i]:
                sj = ssect_2_sect[j]
                reject_out[si,sj] = IS_VISIBLE
                reject_out[sj,si] = IS_VISIBLE
        print(f'PVSs finished: {int(time.perf_counter() - tt)} sec')

    write_reject(reject_out, OUT_REJECT)

    #
    if PLOT_REJECT:
        fig = mpl.figure(0, figsize=(10,10), dpi=200)
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
                my_segs = segs_list[ssect[1]:ssect[1]+ssect[0]]
                if ssi == ssect_to_plot:
                    for si,seg in enumerate(my_segs):
                        segs_to_plot_copy.append([[seg[0], seg[1]], [0,0,0,1], 2.0])
                if ssi in pvs_result[ssect_to_plot]:
                    for si,seg in enumerate(my_segs):
                        if ssi != ssect_to_plot:
                            segs_to_plot_copy.append([[seg[0], seg[1]], [1,0,0,1], 1.0])
            #
            fig = mpl.figure(1, figsize=(10,10), dpi=200)
            lines  = [n[0] for n in segs_to_plot_copy]
            clines = [n[1] for n in segs_to_plot_copy]
            widths = [n[2] for n in segs_to_plot_copy]
            lc = mc.LineCollection(lines, colors=clines, linewidths=widths)
            mpl.gca().add_collection(lc)
            mpl.axis('scaled')
            #mpl.show()
            mpl.savefig(f'{OUT_REJECT}.{ssect_to_plot}.png')
            mpl.close(fig)
        print(f'plotting finished: {int(time.perf_counter() - tt)} sec')


if __name__ == '__main__':
    main()
