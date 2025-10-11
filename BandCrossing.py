from pymatgen.electronic_structure.bandstructure import BandStructure, BandStructureSymmLine, Kpoint
from pymatgen.core.lattice import Lattice
from pymatgen.analysis.interfaces import ZSLGenerator
from pymatgen.analysis.interfaces import ZSLMatch
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.core.structure import Structure
from pymatgen.electronic_structure.core import Spin, Orbital
import numpy as np
from itertools import product

import Band_Match

def band_overlap(band1_o,band2_o,dif_vac):
    #get band structure
    band1 = band1_o
    band2 = band2_o
    #adjust energy values w.r.t. new fermi level
    for key,value in band1.bands.items():
        value1 = value
        band1.bands[key] = value1
    for key,value in band2.bands.items():
        value1 = value + dif_vac
        band2.bands[key] = value1

    kpoints1 = band1.kpoints
    rlat1 = band1.lattice_rec
    kpoints2 = band2.kpoints
    rlat2 = band2.lattice_rec
    lat1 = band1.structure.lattice
    lat2 = band2.structure.lattice

    #compute supercells
    a1 = lat1.matrix[0]
    a2 = lat1.matrix[1]
    b1 = lat2.matrix[0]
    b2 = lat2.matrix[1]
    film_area = np.linalg.norm(np.cross(a1,a2))
    substrate_area = np.linalg.norm(np.cross(b1,b2))
    Match_gen = ZSLGenerator(max_area_ratio_tol=0.1, max_area=1.1*film_area, max_length_tol=0.1, max_angle_tol=0.1)
    transformation = Match_gen.generate_sl_transformation_sets(film_area, substrate_area)
    film_vectors = np.array([a1,a2])
    substrate_vectors = np.array([b1,b2])
    supercell_generator = Match_gen.get_equiv_transformations(transformation, film_vectors, substrate_vectors)
    for matches in supercell_generator:
        film_sl_vectors = matches[0]
        substrate_sl_vectors = matches[1]
        Tm1 = matches[2]
        Tm2 = matches[3]
        Match_post = ZSLMatch(film_sl_vectors, substrate_sl_vectors, film_vectors, substrate_vectors, Tm1, Tm2)
        Tmu = Match_post.match_transformation
    #acquire the transformation matrices
    Tm1 = np.pad(Tm1,((0,1),(0,1)),'constant', constant_values=0)
    Tm2 = np.pad(Tm2,((0,1),(0,1)),'constant', constant_values=0)
    Tm1[2][2] = 1
    Tm2[2][2] = 1

    #mini Brilloun zone
    Tm1p = np.transpose(np.linalg.inv(Tm1))
    Tm2p = np.transpose(np.linalg.inv(Tm2))
    Tmup = np.transpose(np.linalg.inv(Tmu))
    rlat1sm = np.transpose(np.matmul(Tm1p,np.transpose(rlat1.matrix)))
    rlat2sm = np.transpose(np.matmul(Tm2p,np.transpose(rlat2.matrix)))
    rlat1s = Lattice(rlat1sm)
    rlat2s = Lattice(rlat2sm)
    lat1sm = np.transpose(np.matmul(Tm1,np.transpose(lat1.matrix)))
    lat2sm = np.transpose(np.matmul(Tm1,np.transpose(lat2.matrix)))
    lat1s = Lattice(lat1sm)
    lat2s = Lattice(lat2sm)
    areas = np.linalg.norm(np.cross(lat1sm[0],lat1sm[1]))
    area_film = np.linalg.norm(np.cross(a1,a2))
    area_subs = np.linalg.norm(np.cross(b1,b2))
    size_ratiof = round(areas/area_film)
    size_ratios = round(areas/area_subs)

    band1_dict = band1.as_dict()['bands']
    band2_dict = band2.as_dict()['bands']

    #get the folded band structure, if the supercell size is at the same size, then skip it.
    terms_inter = 3
    kpath1 = [[kpoints1[i].frac_coords[0],kpoints1[i].frac_coords[1],kpoints1[i].frac_coords[2]] for i in range(len(kpoints1))]
    kpath2 = [[kpoints2[i].frac_coords[0],kpoints2[i].frac_coords[1],kpoints2[i].frac_coords[2]] for i in range(len(kpoints2))]
    keys_list = list(band1_dict.keys())
    band1_hsl = {key: [[] for _ in range(len(kpath1))] for key in keys_list}
    for key in keys_list:
        band_num = len(band1_dict[key])
        for i in range(len(kpath1)):
            en_to_append = []
            for j in range(band_num):
                en_to_append.append(band1_dict[key][j][i])
            band1_hsl[key][i] = en_to_append

    band2_hsl = {key: [[] for _ in range(len(kpath1))] for key in keys_list}
    for key in keys_list:
        band_num = len(band2_dict[key])
        for i in range(len(kpath2)):
            en_to_append = []
            for j in range(band_num):
                en_to_append.append(band2_dict[key][j][i])
            band2_hsl[key][i] = en_to_append
            
    if size_ratiof > 1:
        band1_new,kdense1 = Band_Match.interp_hsl_hex(kpoints1, band1_dict, terms_inter)
        #get space group from structure
        sg1 = SpacegroupAnalyzer(band1.structure)
        sg1_op = sg1.get_point_group_operations(cartesian=True)
        #apply operations in space group to k points
        k1_tot,band1_dict_rot = Band_Match.point_group(sg1_op,kdense1,rlat1,band1_new)
        #find out points in BZ related to the high symmetry line in the superBZ
        band1_hsl = Band_Match.fold_band(kpoints1,rlat1,size_ratiof,rlat1sm,k1_tot,band1_dict_rot)

    if size_ratios > 1:
        band2_new,kdense2 = Band_Match.interp_hsl_hex(kpoints2, band2_dict, terms_inter)
        sg2 = SpacegroupAnalyzer(band1.structure)
        sg2_op = sg2.get_point_group_operations(cartesian=True)
        k2_tot,band2_dict_rot = Band_Match.point_group(sg2_op,kdense2,rlat2,band2_new)
        band2_hsl = Band_Match.fold_band(kpoints2,rlat2,size_ratios,rlat2sm,k2_tot,band2_dict_rot)

    return band1_hsl,band2_hsl

def overlap_ind_old_2dMat(new_fermi,DelEv,band1_hsl,band2_hsl,kpath1,en_range_width=1):
    #put two bands together
    #energy range to consider 
    en_range = [new_fermi - en_range_width/2, new_fermi + en_range_width/2]
    keys_list = list(band1_hsl.keys())
    overlap = []
    count = 0
    for i in range(len(band1_hsl[keys_list[0]])):#i runs over momenta
        if i > 0:
            if kpath1[i] == kpath1[i-1]:
                count += 1
                continue
        en_tobecom1 = []
        en_tobecom2 = []
        #combine energies of two spins
        for key in keys_list:
            for j in range(len(band1_hsl[key][i])):
                if band1_hsl[key][i][j] > en_range[0] and band1_hsl[key][i][j] < en_range[1]:
                    en_tobecom1.append(band1_hsl[key][i][j])
                if band2_hsl[key][i][j] > en_range[0] and band2_hsl[key][i][j] < en_range[1]:
                    en_tobecom2.append(band2_hsl[key][i][j]+DelEv)
        #compute how many overlaps for one momentum
        overlap_temp = 0

        for en1 in en_tobecom1:
            en_dif = np.array(en_tobecom2) - en1
            #the overlap indices equals to the exponential of difference
            overlap_temp += sum([np.exp(- np.absolute(en/en_range_width)) for en in en_dif])
        overlap.append(overlap_temp)

    overlap_index = sum(overlap)
    return overlap, overlap_index

def overlap_ind_old(new_fermi,DelEv,band1_hsl,band2_hsl,kpath1,en_range_width=1):
    #put two bands together
    #energy range to consider 
    en_range = [new_fermi - en_range_width/2, new_fermi + en_range_width/2]
    keys_list = list(band1_hsl.keys())
    overlap = []
    count = 0
    for i in range(len(band1_hsl[keys_list[0]][0])):#i runs over momenta
        if i > 0:
            if kpath1[i] == kpath1[i-1]:
                count += 1
                continue
        en_tobecom1 = []
        en_tobecom2 = []
        #combine energies of two spins
        for key in keys_list:
            for j in range(len(band1_hsl[key])):
                if band1_hsl[key][j][i] > en_range[0] and band1_hsl[key][j][i] < en_range[1]:
                    en_tobecom1.append(band1_hsl[key][i][j])
                if band2_hsl[key][j][i] > en_range[0] and band2_hsl[key][j][i] < en_range[1]:
                    en_tobecom2.append(band2_hsl[key][i][j]+DelEv)
        #compute how many overlaps for one momentum
        overlap_temp = 0

        for en1 in en_tobecom1:
            en_dif = np.array(en_tobecom2) - en1
            #the overlap indices equals to the exponential of difference
            overlap_temp += sum([np.exp(- np.absolute(en/en_range_width)) for en in en_dif])
        overlap.append(overlap_temp)

    overlap_index = sum(overlap)
    return overlap, overlap_index

def overlap_ind_hsl(new_fermi,DelEv,band1_hsl,band2_hsl,kpath,en_range_width=1,num_slices=5,projection=True):
    #if bands are projected by spin
    #put two bands together
    #energy range to consider
    en_range = [new_fermi - en_range_width/2, new_fermi + en_range_width/2]
    overlap = np.zeros((num_slices,len(kpath)))
    slices = np.linspace(en_range[0],en_range[1],num_slices+1)
    count = 0

    for i in range(len(kpath)):#i runs over momenta
        en_tobecom1 = []
        en_tobecom2 = []
        if i > 0:
            if kpath[i] == kpath[i-1]:
                count += 1
                continue
        if projection == True:
            keys_list = list(band1_hsl.keys())
            #combine energies of two spins
            for key in keys_list:
                for j in range(len(band1_hsl[key])):
                    if band1_hsl[key][j][i] >= en_range[0] and band1_hsl[key][j][i] < en_range[1]:
                        en_tobecom1.append(band1_hsl[key][j][i])
                for j in range(len(band2_hsl[key])):
                    if band2_hsl[key][j][i] >= en_range[0] and band2_hsl[key][j][i] < en_range[1]:
                        en_tobecom2.append(band2_hsl[key][j][i]+DelEv)
                        
        else:   
            #combine energies of two spins
            for j in range(len(band1_hsl)):
                if band1_hsl[j][i] >= en_range[0] and band1_hsl[j][i] < en_range[1]:
                    en_tobecom1.append(band1_hsl[j][i])
            for k in range(len(band2_hsl)):
                if band2_hsl[k][i] >= en_range[0] and band2_hsl[k][i] < en_range[1]:
                    en_tobecom2.append(band2_hsl[k][i]+DelEv)
    
        #compute how many overlaps for one momentum
        overlap_temp = 0
        #do comparison in every small slice of energy range
        for t in range(num_slices):
            #first search for eigenvalues inside small ranges
            sm_ran = [slices[t], slices[t+1]]
            en_slice1 = []
            for en in en_tobecom1:
                if en >= sm_ran[0] and en < sm_ran[1]:
                    en_slice1.append(en)
            en_slice2 = []
            for en in en_tobecom2:
                if en >= sm_ran[0] and en < sm_ran[1]:
                    en_slice2.append(en)
            #do comparison
            for en1 in en_slice1:
                en_dif = np.array(en_slice2) - en1
                #the overlap indices equals to the exponential of difference
                overlap_temp += sum([np.exp(- np.sqrt(np.absolute(en * num_slices/en_range_width))) for en in en_dif])
            overlap[t,i] = overlap_temp

    overlap_slice = np.sum(overlap, axis=1)
    overlap_index = max(overlap_slice)
    return overlap, overlap_index

def coarse_k(kpoints,density_k):
    """
    generate coarse k grid from dense k grid 

    return coarse_k_grid, assigned_kpoints: (density_k ** 2, indefinite number) corresponding positions 
    in the list of kpoints for every point in coarse_k_grid
    
    """
    k_min_1 = min(kpoints, key=lambda k: k[0])
    k_min_2 = min(kpoints, key=lambda k: k[1])
    k_max_1 = max(kpoints, key=lambda k: k[0])
    k_max_2 = max(kpoints, key=lambda k: k[1])
    k1_sp = np.linspace(k_min_1[0], k_max_1[0], num=density_k)
    k2_sp = np.linspace(k_min_2[1], k_max_2[1], num=density_k)
    k_pool = [[k1, k2, 0.] for k1, k2 in product(k1_sp, k2_sp)]
    k_assign = np.empty((len(k_pool),),dtype=object)
    for i in range(len(k_assign)):
        k_assign[i] = []
    #find k points in k_pool that have nearby k points existed in kpoints
    for i in range(len(kpoints)):
        ind = np.argmin(np.abs(k1_sp - kpoints[i][0])) * density_k + np.argmin(np.abs(k2_sp - kpoints[i][1]))
        k_assign[ind].append(i)
    return k_pool, k_assign


def overlap_ind_kgrid(new_fermi, DelEv, band1, band2, kpoints1, kpoints2, en_range_width=1,
                      num_slices=5, projection=True, density_k=8):
    #if bands are projected by spin
    #put two bands together
    #energy range to consider
    en_range = [new_fermi - en_range_width/2, new_fermi + en_range_width/2]
    slices = np.linspace(en_range[0],en_range[1],num_slices+1)
    #define k_pool to the kpoints as labels of eigenvalues, reduce number of k points to density_k ** 2
    k_pool, k_assign1 = coarse_k(kpoints1, density_k)
    k_pool, k_assign2 = coarse_k(kpoints2, density_k)
    overlap = np.zeros((num_slices, len(k_pool)))

    for i in range(len(k_pool)):#i runs over momenta
        en_tobecom1 = []
        en_tobecom2 = []
        k_pos1 = k_assign1[i]
        k_pos2 = k_assign2[i]
        if projection:
            keys_list = list(band1.keys())
            #combine energies of two spins
            for key in keys_list:
                for j in range(len(band1[key])):
                    for k in k_pos1:
                        if en_range[0] <= band1[key][j][k] < en_range[1]:
                            en_tobecom1.append(band1[key][j][k])
                    for k in k_pos2:
                        if en_range[0] <= band2[key][j][k] < en_range[1]:
                            en_tobecom2.append(band2[key][j][k] + DelEv)
                            
        else:   
            #combine energies of two spins
            for j in range(len(band1)):
                for k in k_pos1:
                    if en_range[0] <= band1[j][k] < en_range[1]:
                        en_tobecom1.append(band1[j][k])
            for j in range(len(band2)):    
                for k in k_pos2:
                    if en_range[0] <= band2[j][k] < en_range[1]:
                        en_tobecom2.append(band2[j][k] + DelEv)
        #compute how many overlaps for one momentum
        overlap_temp = 0
        #do comparison in every small slice of energy range
        for t in range(num_slices):
            #first search for eigenvalues inside small ranges
            sm_ran = [slices[t], slices[t+1]]
            en_slice1 = []
            for en in en_tobecom1:
                if sm_ran[0] <= en < sm_ran[1]:
                    en_slice1.append(en)
            en_slice2 = []
            for en in en_tobecom2:
                if sm_ran[0] <= en < sm_ran[1]:
                    en_slice2.append(en)
            #do comparison
            for en1 in en_slice1:
                en_dif = np.array(en_slice2) - en1
                #the overlap indices equals to the exponential of difference
                overlap_temp += sum([np.exp(- np.sqrt(np.absolute(en * num_slices/en_range_width))) for en in en_dif])
            overlap[t,i] = overlap_temp

    overlap_slice = np.sum(overlap, axis=1)
    overlap_index = max(overlap_slice)
    return overlap, overlap_index
