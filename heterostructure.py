'''
Functions used to generate heterostructures
'''
import os
from jarvis.io.vasp.inputs import Poscar
from jarvis.core.atoms import Atoms

# from jarvis.analysis.interface.zur import make_interface
# import importlib
# importlib.reload(jarvis_interface)
from jarvis_interface import make_interface, ZSLGenerator
from pymatgen.core.structure import Structure
from pymatgen.core.lattice import Lattice
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from jarvis.core.atoms import pmg_to_atoms
import Elastic
import pandas as pd
import numpy as np
import contextlib

# import pymatgen.core.interface
# from pymatgen.analysis.interfaces import zsl
# from pyxtal import pyxtal
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core.operations import SymmOp
from pymatgen.core.periodic_table import Element
import ast
import heterostructure


def Rot_Center(film, subs, index_atom_film, index_atom_subs):
    film = film.center_around_origin([0, 0, 0])
    subs = subs.center_around_origin([0, 0, 0])
    film_py = film.pymatgen_converter()
    subs_py = subs.pymatgen_converter()
    subs_sym = SpacegroupAnalyzer(subs_py).get_symmetry_dataset()
    film_sym = SpacegroupAnalyzer(film_py).get_symmetry_dataset()
    origin_film = film_sym["origin_shift"]
    origin_subs = subs_sym["origin_shift"]
    # origin_film = -film_crystal.atom_sites[index_atom_film].position[:2] - film.frac_coords[index_atom_film][:2]
    # origin_subs = -subs_crystal.atom_sites[index_atom_subs].position[:2] - subs.frac_coords[index_atom_subs][:2]
    # fractional distances between input fractional coordinates of atoms and standard coordinates of maximal wycoff positions
    r_film = [origin_film[0], origin_film[1], 0.0]
    r_subs = [origin_subs[0], origin_subs[1], 0.0]
    return r_film, r_subs


def DISPLACEMENT_wyckoff_list(film, subs, index_atom_film, index_atom_subs):
    displacement_list = [np.zeros(3)]
    displacement_list.append(
        -subs.lattice.frac_coords(subs.cart_coords[index_atom_subs])
    )
    displacement_list.append(
        subs.lattice.frac_coords(film.cart_coords[index_atom_subs])
    )
    displacement_list.append(
        -subs.lattice.frac_coords(
            subs.cart_coords[index_atom_subs] - film.cart_coords[index_atom_film]
        )
    )
    for dis in displacement_list:
        dis[2] = 0.0
    return displacement_list


def DISPLACEMENT_wyckoff(film, subs, index_atom_film, index_atom_subs):
    displacement = subs.lattice.frac_coords(
        subs.cart_coords[index_atom_subs] - film.cart_coords[index_atom_film]
    )
    displacement[2] = 0.0
    return displacement


# apply flipping function
def Apply_flipping(Atom):
    Atom_list = [Atom]
    Atom_py = Atom.pymatgen_converter()
    flip_matrix = np.array(
        [
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    flip_ops = SymmOp(flip_matrix)
    Atom_after_flipping = Atom_py.apply_operation(flip_ops)
    # if not Atom_after_flipping.matches(Atom):
    Atom_list.append(pmg_to_atoms(Atom_after_flipping))
    return Atom_list


def Apply_Rotation(Atom):
    if isinstance(Atom, list):
        Atom_list = Atom
    else:
        Atom_list = [Atom]
    Atom_py = Atom.pymatgen_converter()
    Atom_list_py = [Atom_py]
    lattice_dict = Atom_py.lattice.matrix
    lattH = Structure(lattice_dict, ["H"], [[0, 0, 0]])
    spg_Atom_lattice = SpacegroupAnalyzer(lattH)  # space group of lattice structure
    spg_Atom = SpacegroupAnalyzer(
        Atom_py
    )  # space group of film with atomic information

    # apply rotations, use pymatgen symmetry module
    symm_ops_latt = spg_Atom_lattice.get_point_group_operations(cartesian=True)
    symm_ops_Atom = spg_Atom.get_point_group_operations(cartesian=True)
    symm_number = 0
    for iop, ops_latt in enumerate(symm_ops_latt):
        if iop > 0:
            rot = ops_latt.rotation_matrix
            if (
                np.abs(np.linalg.det(rot) - 1) < 1e-4 and np.abs(rot[2][2] - 1) < 1e-4
            ):  # to tell if it is a rotation
                compare_indicator = 1
                # to exclude operators in the symmetry group of film
                for ops in symm_ops_Atom:
                    rot_Atom = ops.rotation_matrix
                    if (
                        np.abs(np.linalg.det(rot_Atom) - 1) < 1e-4
                        and np.abs(rot_Atom[2][2] - 1) < 1e-4
                    ):  # to tell if it is a rotation
                        if (np.abs(rot_Atom - rot) < 1e-4).all():
                            compare_indicator = 0
                            break

                if compare_indicator == 1:
                    symm_number += 1
                    Atom_after_rot = Atom_py.apply_operation(ops_latt)
                    Atom_list_py.append(Atom_after_rot)
    for i, atom_rot_py1 in enumerate(Atom_list_py):
        identical_structure_indicator = 1
        for j, atom_rot_py2 in enumerate(Atom_list_py):
            if j > i:
                if atom_rot_py1.matches(atom_rot_py2):
                    identical_structure_indicator = 0
                    break
        if identical_structure_indicator == 1:
            atom_rot = pmg_to_atoms(atom_rot_py1)
            Atom_list.append(atom_rot)
    return Atom_list


# find translation vectors in order to generate different stackings
def Translation_List(film, subs):
    t_list = [np.array([0.0, 0.0, 0.0])]
    t_list_uniq = []
    film_coords = np.array(film.cart_coords)
    subs_coords = np.array(subs.cart_coords)
    all_coords = np.concatenate((film_coords, subs_coords), axis=0)
    for i in range(len(film_coords)):
        for j in range(i, len(film_coords)):
            t_i = film_coords[i]
            t_j = film_coords[j]
            t_ij0 = t_i - t_j
            t_ij = np.array([t_ij0[0], t_ij0[1], 0.0])
            if (np.abs(t_ij) > 1e-4).any():
                t_frac = subs.lattice.frac_coords(t_ij)
                t_list.append(t_frac)
                t_list.append(-t_frac)
    for i in range(len(subs_coords)):
        for j in range(i, len(subs_coords)):
            t_i = subs_coords[i]
            t_j = subs_coords[j]
            t_ij0 = t_i - t_j
            t_ij = np.array([t_ij0[0], t_ij0[1], 0.0])
            if (np.abs(t_ij) > 1e-4).any():
                t_frac = subs.lattice.frac_coords(t_ij)
                t_list.append(t_frac)
                t_list.append(-t_frac)

    """
    
    for i in range(len(film.cart_coords)):
        for j in range(len(film.cart_coords)):
            if j > i:
                t_i = film.cart_coords[i][:2]
                t_j = film.cart_coords[j][:2]
                t_ij = [(t_i - t_j)[0],(t_i - t_j)[1],0]
                if (np.abs(t_ij) > 1e-4).any():
                    t_frac = subs.lattice.frac_coords(t_ij)
                    t_list.append(t_frac)
    for i in range(len(subs.cart_coords)):
        for j in range(len(subs.cart_coords)):
            if j > i:
                t_i = subs.cart_coords[i][:2]
                t_j = subs.cart_coords[j][:2]
                t_ij = [(t_i - t_j)[0],(t_i - t_j)[1],0]
                if (np.abs(t_ij) > 1e-4).any():
                    t_frac = subs.lattice.frac_coords(t_ij)
                    t_list.append(t_frac)
    """
    # check redundency
    for i, t1 in enumerate(t_list):
        identical_structure_indicator = 1
        for j, t2 in enumerate(t_list):
            if j > i:
                if (np.abs(t1 - t2) < 1e-3).all():
                    identical_structure_indicator = 0
                    break
        if identical_structure_indicator == 1:
            t_list_uniq.append(t1)
    return t_list_uniq


# make a list of heterostructure from film, substrate and list of translations
def Make_Het_Translation(
    film, subs, r_film, r_subs, t_list, displacement=0, seperation=4, vacuum=20
):
    hetero_list = []
    film = remove_vac(film)
    subs = remove_vac(subs)
    for t_ij in t_list:
        hetero_eq = make_interface(
            film=film,
            subs=subs,
            r_film=r_film,
            r_subs=r_subs,
            seperation=seperation,
            vacuum=vacuum,
            shift=displacement + t_ij,
        )
        hetero_list.append(hetero_eq)

    return hetero_list


def exclude_duplicate(het_list):
    matcher = StructureMatcher(ltol=0.05, stol=0.05, angle_tol=5, scale=False)
    het_list_uniq = []
    atoms_py = [
        het["interface"].center_around_origin().pymatgen_converter() for het in het_list
    ]
    for i, atom1 in enumerate(atoms_py):
        identical_structure_indicator = 1
        for j, atom2 in enumerate(atoms_py):
            if j > i:
                atom1 = atoms_py[i]
                atom2 = atoms_py[j]
                if matcher.fit(atom1, atom2):
                    identical_structure_indicator = 0
                    break
        if identical_structure_indicator == 1:
            het_list_uniq.append(het_list[i])
    return het_list_uniq


def remove_vac(atom1, keep_vac=3.0):
    top_z = max(np.array(atom1.cart_coords)[:, 2])
    bot_z = min(np.array(atom1.cart_coords)[:, 2])
    lat_z = [0.0, 0.0, top_z - bot_z + 1e-2 + keep_vac]
    lat_mat = [atom1.lattice_mat[0], atom1.lattice_mat[1], lat_z]
    new_atoms = Atoms(
        lattice_mat=lat_mat,
        coords=atom1.cart_coords,
        elements=atom1.elements,
        cartesian=True,
    )
    # print(new_atoms)
    frac_coords = new_atoms.frac_coords
    translation = [0.0, 0.0, round(frac_coords[0, 2])]
    frac_coords[:] = frac_coords[:] - translation
    atom1_n = Atoms(
        lattice_mat=lat_mat,
        coords=frac_coords,
        elements=atom1.elements,
        cartesian=False,
    )
    return atom1_n


def gap_adjust(het_list, r_film, r_subs):
    het_list_gap = []
    for shift_i, het in enumerate(het_list):
        shift = het["shift"]
        top = het["film_sl"].center_around_origin(r_film)
        bottom = het["subs_sl"].center_around_origin(r_subs + shift)
        top_py = top.pymatgen_converter()
        bottom_py = bottom.pymatgen_converter()
        lat_vec1 = bottom.lattice_mat[0]
        lat_vec2 = bottom.lattice_mat[1]
        a1 = np.linalg.norm(lat_vec1)
        a2 = np.linalg.norm(lat_vec2)
        a_crit = max(a1, a2)

        num_top = len(top_py.species)
        num_bottom = len(bottom_py.species)
        top_vdw = [spec.van_der_waals_radius for spec in top_py.species]
        bottom_vdw = [spec.van_der_waals_radius for spec in bottom_py.species]
        # maximum distance to start iteration
        max_dz = 2 * max(np.max(top_vdw), np.max(bottom_vdw)) + 0.5
        dz_arr = np.linspace(10.0, 2.0, 80)
        dz_best = 3  # initial value
        for dz in dz_arr:
            # for every distance, compute distances minus sum of van der Waal radius (normalized), until one of those values reaches zero
            dis_van_cross = np.zeros((num_bottom, num_top))
            for i in range(num_bottom):
                for j in range(num_top):
                    coord1 = bottom_py.cart_coords[i]
                    coord2 = top_py.cart_coords[j] + [0.0, 0.0, dz]
                    distance = np.linalg.norm(coord1 - coord2)
                    # get reduced distance (consider lattice translation)
                    if distance > a_crit:
                        distance_list = [distance]
                        for l in [-2.0, -1.0, 0.0, 1.0, 2.0]:
                            for n in [-2.0, -1.0, 0.0, 1.0, 2.0]:
                                coord1_temp = coord1 + l * lat_vec1 + n * lat_vec2
                                distance_temp = np.linalg.norm(coord1_temp - coord2)
                                distance_list.append(distance_temp)
                        distance = np.min(distance_list)

                    sum_vdw = bottom_vdw[i] + top_vdw[j]
                    dis_van = distance - sum_vdw
                    dis_van_cross[i, j] = dis_van

            if np.min(dis_van_cross) < 0:
                dz_best = dz
                break
        # het_list_gap.append(make_interface(film=top, subs=bottom, r_film=r_film, r_subs = r_subs, seperation = dz_best - 0.5 , vacuum = 20,shift=het['shift']))
        # het_list_gap.append(make_interface(film=top, subs=bottom, r_film=r_film, r_subs = r_subs, seperation = dz_best - 0.25, vacuum = 20,shift=het['shift']))
        het_list_gap.append(
            make_interface(
                film=top,
                subs=bottom,
                r_film=r_film,
                r_subs=r_subs,
                seperation=dz_best,
                vacuum=20,
                shift=het["shift"],
            )
        )
        # het_list_gap.append(make_interface(film=top, subs=bottom, r_film=r_film, r_subs = r_subs, seperation = dz_best + 1., vacuum = 20,shift=het['shift']))
        # het_list_gap.append(make_interface(film=top, subs=bottom, r_film=r_film, r_subs = r_subs, seperation = dz_best + 2., vacuum = 20,shift=het['shift']))
    return het_list_gap


def elastic_tensor(ori_ela: dict):
    elastic_tensor = [
        [ori_ela["c_11"], ori_ela["c_12"], 0, 0, 0, ori_ela["c_13"] / np.sqrt(2)],
        [ori_ela["c_21"], ori_ela["c_22"], 0, 0, 0, ori_ela["c_23"] / np.sqrt(2)],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [
            ori_ela["c_31"] / np.sqrt(2),
            ori_ela["c_32"] / np.sqrt(2),
            0,
            0,
            0,
            ori_ela["c_33"] / 2,
        ],
    ]
    return elastic_tensor


def remove_bad_stack(het_list):
    good_stack = []
    for het in het_list:
        het_atom = het["interface"].center_around_origin()
        if_good = 0
        for i in het_atom.cart_coords[:6]:
            for j in het_atom.cart_coords[6:]:
                if np.max(np.abs(i[:2] - j[:2])) < 1e-4:
                    if_good = 1
        if if_good == 1:
            good_stack.append(het)
    return good_stack
