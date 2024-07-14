#!/usr/bin/env python
import argparse
import copy
import numpy as np
import time

from concurrent.futures import ProcessPoolExecutor
from itertools import repeat

from source.pvs_func import precompute_portal_visibility, PVS_DFS
from source.wad_func import *

PPV_BATCHSIZE = 50


def main(raw_args=None):
    parser = argparse.ArgumentParser(description='pvs-reject', formatter_class=argparse.ArgumentDefaultsHelpFormatter,)
    parser.add_argument('-i', type=str, required=True,  metavar='input.wad',  help="* Input WAD")
    parser.add_argument('-m', type=str, required=True,  metavar='MAP01',      help="* Map name")
    parser.add_argument('-r', type=str, required=True,  metavar='REJECT.lmp', help="* Output reject table lmp")
    parser.add_argument('-v', type=str, required=False, metavar='vis.npz',    help="Load precomputed visibility matrix", default='')
    parser.add_argument('-p', type=int, required=False, metavar='4',          help="Number of processes to use", default=4)
    parser.add_argument('--save-vis',   required=False, action='store_true',  help="Save precomputed visibility matrix", default=False)
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
    SAVE_VISIBILITY = args.save_vis
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

    ssect_graph = [[] for n in range(n_subsectors)]
    for i in range(portal_ssects.shape[0]):
        ssect_graph[portal_ssects[i,0]].append((portal_ssects[i,1], i))

    sect_graph = [[] for n in range(n_sectors)]
    for i in range(portal_ssects.shape[0]):
        my_sector = ssect_2_sect[portal_ssects[i,0]]
        partner_sector = ssect_2_sect[portal_ssects[i,1]]
        if my_sector != partner_sector:
            sect_graph[my_sector].append((partner_sector, i))

    if LOAD_VISIBILITY:
        print('loading precomputed visibility from file...')
        in_npz = np.load(LOAD_VISIBILITY)
        portal_cantsee = in_npz['portal_cantsee']
    else:
        tt = time.perf_counter()
        portal_cantsee = np.zeros(((n_portals*n_portals)//8 + 1), dtype='B')
        for ssi_start in range(0, n_subsectors, PPV_BATCHSIZE):
            my_ssi = range(ssi_start, min(ssi_start+PPV_BATCHSIZE, n_subsectors))
            with ProcessPoolExecutor(max_workers=NUM_PROCESSES) as executor:
                cantsee_result = list(executor.map(precompute_portal_visibility, repeat(ssect_graph), repeat(portal_coords), my_ssi, repeat(PRINT_PROGRESS)))
            for ib_list in cantsee_result:
                for i in range(0,len(ib_list),2):
                    portal_cantsee[ib_list[i]] |= ib_list[i+1]
            del cantsee_result
        if SAVE_VISIBILITY:
            np.savez_compressed(f'{OUT_REJECT}.npz', portal_cantsee=portal_cantsee)
        print(f'portal visibility precomputation finished: {int(time.perf_counter() - tt)} sec')

    ####tt = time.perf_counter()
    ####with ProcessPoolExecutor(max_workers=NUM_PROCESSES) as executor:
    ####    pvs_result = list(executor.map(PVS_DFS, repeat(sect_graph), repeat(portal_coords), range(n_sectors), repeat(portal_cantsee), repeat(PRINT_PROGRESS)))
    ####reject_out = np.zeros((n_sectors, n_sectors), dtype='bool') + IS_INVISIBLE
    ####for si in range(n_sectors):
    ####    for sj in pvs_result[si]:
    ####        reject_out[si,sj] = IS_VISIBLE
    ####        reject_out[sj,si] = IS_VISIBLE
    ####print(f'PVSs finished: {int(time.perf_counter() - tt)} sec')

    tt = time.perf_counter()
    with ProcessPoolExecutor(max_workers=NUM_PROCESSES) as executor:
        pvs_result = list(executor.map(PVS_DFS, repeat(ssect_graph), repeat(portal_coords), range(n_subsectors), repeat(portal_cantsee), repeat(PRINT_PROGRESS)))
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
