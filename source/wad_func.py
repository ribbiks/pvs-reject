import numpy as np
from struct import unpack

BITMASKS = [0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80]
IS_VISIBLE = 0
IS_INVISIBLE = 1


def null_pad(string, length=8):
    return string + chr(0) * (length - len(string))


MAP_LUMPS = ['THINGS', 'LINEDEFS', 'SIDEDEFS', 'VERTEXES', 'SEGS', 'SSECTORS', 'NODES', 'SECTORS', 'REJECT', 'BLOCKMAP',
             'GL_VERT', 'GL_SEGS', 'GL_SSECT', 'GL_NODES']
MAP_LUMPS = {null_pad(n):True for n in MAP_LUMPS}


# 2d numpy array --> reject lump
def write_reject(reject, fn):
    bytes_out = []
    total_size = reject.shape[0]*reject.shape[1]
    reject = reject.reshape((total_size))
    reject = np.hstack((reject, np.zeros((8), dtype='<i4')))
    for i in range(0,total_size,8):
        my_byte = 1*reject[i+7] + 2*reject[i+6] + 4*reject[i+5] + 8*reject[i+4] + 16*reject[i+3] + 32*reject[i+2] + 64*reject[i+1] + 128*reject[i]
        bytes_out.append(my_byte)
    bytes_out = bytes(bytes_out)
    with open(fn,'wb') as f:
        f.write(bytes_out)


def get_map_lmps(in_wad, which_map):
    which_map = null_pad(which_map)
    wad_data = []
    lmp_list = []
    with open(in_wad,'rb') as f:
        f.read(4) # PWAD
        (n_lmp, p_dir) = unpack('ii', f.read(8))
        f.seek(p_dir)
        for i in range(n_lmp):
            (lmp_pos, lmp_size) = unpack('ii', f.read(8))
            lmp_name = f.read(8)
            lmp_list.append((lmp_name, lmp_pos, lmp_size))
        for n in lmp_list:
            f.seek(n[1])
            wad_data.append((n[0], f.read(n[2])))
    #
    map_data = {}
    in_current_map = False
    for n in wad_data:
        lmp_name = n[0].decode("utf-8")
        if lmp_name == which_map:
            in_current_map = True
            MAP_LUMPS['GL_' + which_map[:5]] = True
        elif in_current_map and lmp_name in MAP_LUMPS:
            map_data[lmp_name] = n[1]
        else:
            in_current_map = False
    #
    return map_data


def get_linedefs(map_data):
    line_list = []
    line_data = map_data[null_pad('LINEDEFS')]
    for offset in range(0,len(line_data),14):
        (start_vertex, end_vertex, line_flags, line_special, line_tag, sidedef_front, sidedef_back) = unpack('HHhhhHH', line_data[offset:offset+14])
        line_list.append((start_vertex, end_vertex, line_flags, line_special, line_tag, sidedef_front, sidedef_back))
    return line_list


def get_sidedefs(map_data):
    side_list = []
    side_data = map_data[null_pad('SIDEDEFS')]
    for offset in range(0,len(side_data),30):
        (x_off, y_off) = unpack('hh', side_data[offset:offset+4])
        upper_tex = side_data[offset+4:offset+12]
        lower_tex = side_data[offset+12:offset+20]
        middle_tex = side_data[offset+20:offset+28]
        facing_sector = unpack('h', side_data[offset+28:offset+30])[0]
        side_list.append((x_off, y_off, upper_tex, lower_tex, middle_tex, facing_sector))
    return side_list


def get_sectors(map_data):
    sect_list = []
    sect_data = map_data[null_pad('SECTORS')]
    for offset in range(0,len(sect_data),26):
        (floor_height, ceil_height) = unpack('hh', sect_data[offset:offset+4])
        floor_flat = sect_data[offset+4:offset+12]
        ceil_flat  = sect_data[offset+12:offset+20]
        (sec_light, sec_special, sec_tag) = unpack('hhh', sect_data[offset+20:offset+26])
        sect_list.append((floor_height, ceil_height, floor_flat, ceil_flat, sec_light, sec_special, sec_tag))
    return sect_list


def get_vertexes(map_data):
    normal_verts = []
    vert_data = map_data[null_pad('VERTEXES')]
    for offset in range(0,len(vert_data),4):
        (x, y) = unpack('hh', vert_data[offset:offset+4])
        normal_verts.append((x, y))
    return normal_verts


def get_gl_verts(map_data):
    gl_verts = []
    vert_data = map_data[null_pad('GL_VERT')]
    for offset in range(4,len(vert_data),8):
        (x, y) = unpack('ii', vert_data[offset:offset+8])
        gl_verts.append((x, y))
    return gl_verts


def get_gl_subsectors(map_data):
    ssect_list = []
    ssect_data = map_data[null_pad('GL_SSECT')]
    for offset in range(0,len(ssect_data),8):
        (seg_count, first_seg) = unpack('II', ssect_data[offset:offset+8])
        ssect_list.append((seg_count, first_seg))
    return ssect_list


def get_gl_segs_with_coordinates(map_data, normal_verts, gl_verts):
    segs_list = []
    segs_data = map_data[null_pad('GL_SEGS')]
    for offset in range(0,len(segs_data),16):
        (start_vertex, end_vertex, linedef, side, partner_seg) = unpack('IIHHI', segs_data[offset:offset+16])
        if start_vertex >> 31:
            start_coords = gl_verts[start_vertex - (1 << 31)]
            start_coords = (start_coords[0]/0x10000, start_coords[1]/0x10000)
        else:
            start_coords = normal_verts[start_vertex]
        if end_vertex >> 31:
            end_coords = gl_verts[end_vertex - (1 << 31)]
            end_coords = (end_coords[0]/0x10000, end_coords[1]/0x10000)
        else:
            end_coords = normal_verts[end_vertex]
        segs_list.append((start_coords, end_coords, linedef, side, partner_seg))
    return segs_list


def get_portal_segs(segs_list, ssect_list, line_list, side_list):
    seg_2_sect = {}
    seg_2_ssect = {}
    ssect_2_sect = {}
    segs_to_plot = []
    for ssi,ssect in enumerate(ssect_list):
        my_segs = segs_list[ssect[1]:ssect[1]+ssect[0]]
        my_sectors = {}
        for si,seg in enumerate(my_segs):
            if ssect[1] + si in seg_2_ssect:
                print(f'Seg {ssect[1] + si} associated with multiple subsectors')
                exit(1)
            seg_2_ssect[ssect[1] + si] = ssi
            my_line = seg[2]
            if my_line == 0xffff:
                my_sector = None
            else:
                my_front_side = line_list[my_line][5]
                my_back_side = line_list[my_line][6]
                if seg[3] == 0:
                    my_sector = side_list[my_front_side][5]
                else:
                    my_sector = side_list[my_back_side][5]
            if my_sector is not None:
                my_sectors[my_sector] = True
        if len(my_sectors) == 0:
            print(f'Subsector {ssi} has no sector')
            exit(1)
        elif len(my_sectors) > 1:
            print(f'Subsector {ssi} has multiple sectors')
            exit(1)
        else:
            my_sector = list(my_sectors.keys())[0]
            for si,seg in enumerate(my_segs):
                if ssect[1] + si in seg_2_sect:
                    print(f'Seg {ssect[1] + si} associated with multiple sectors')
                    exit(1)
                seg_2_sect[ssect[1] + si] = my_sector
            ssect_2_sect[ssi] = my_sector
    #
    all_portals = {}
    for ssi,ssect in enumerate(ssect_list):
        my_segs = segs_list[ssect[1]:ssect[1]+ssect[0]]
        my_sector = ssect_2_sect[ssi]
        for si,seg in enumerate(my_segs):
            if seg[4] != 0xffffffff:
                partner_ssect_ind = seg_2_ssect[seg[4]]
                partner_seg = segs_list[seg[4]]
                portal_dat = sorted([[seg[3], ssi, seg[0], seg[1]], [partner_seg[3], partner_ssect_ind, partner_seg[0], partner_seg[1]]])
                portal_dat = (portal_dat[0][1], portal_dat[1][1], portal_dat[0][2], portal_dat[0][3])
                all_portals[portal_dat] = True
                segs_to_plot.append([[seg[0], seg[1]], [0,1,1,1], 0.5])
            else:
                segs_to_plot.append([[seg[0], seg[1]], [0,1,1,1], 0.5])
    all_portals = sorted(all_portals.keys())
    #
    return (all_portals, ssect_2_sect, segs_to_plot)
