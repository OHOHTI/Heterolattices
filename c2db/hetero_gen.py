#use Database C2DB
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
from pathlib import Path

from ase.db import connect
import ase.io
import jarvis_atoms
from ase.io import read
from pymatgen.io.ase import AseAtomsAdaptor
from jarvis.core.atoms import Atoms
from jarvis.core.atoms import pmg_to_atoms

import os
from jarvis.io.vasp.inputs import Poscar
from jarvis_interface import make_interface, ZSLGenerator
import jarvis_interface
import Elastic
import contextlib

# import pymatgen.core.interface
# from pymatgen.analysis.interfaces import zsl
import ast
import heterostructure

missing = 0
uid = []
file_path = []
evac = []
lgnum = []
atoms = []
elastic_tensor = []
label = []
# Example usage
path0 = Path("../data/c2db/c2db")
list_file = "./c2db_step02.xlsx"
for path in path0.glob("A*/*/*"):
    try:
        # Construct the path to the data.json file
        xyz_file = path / "structure.xyz"
        if xyz_file.exists():
            structure_temp = read(xyz_file)
            structure_c2 = AseAtomsAdaptor.get_structure(structure_temp)
            Atom = pmg_to_atoms(structure_c2)

        data_file = path / "data.json"
        # Check if the file exists before reading
        if data_file.exists():
            # Load the JSON data
            data = json.loads(data_file.read_text())

        # record if hse and stiffness json file exist
        hse_file = path / "hse.json"
        ela_file = path / "results-asr.stiffness.json"
        # if hse_file.exists() and ela_file.exists() and data["dyn_stab"] != 'No':
        if ela_file.exists() and data["dyn_stab"] != "No":
            uid_temp = data["uid"]
            evac_temp = data["evac"]
            lgnum_temp = data["lgnum"]
            ela_data = json.loads(ela_file.read_text())
            elastic_tensor_temp = ela_data["kwargs"]["data"]
            uid.append(uid_temp)
            evac.append(evac_temp)
            lgnum.append(lgnum_temp)
            file_path.append(path)
            atoms.append(Atom)
            elastic_tensor.append(elastic_tensor_temp)
            if "label" in data:
                label.append(data["label"])
            else:
                label.append("na")
    except:
        missing += 1

d = {
    "uid": uid,
    "file_path": file_path,
    "evac": evac,
    "lgnum": lgnum,
    "atoms": atoms,
    "elastic_tensor": elastic_tensor,
    "label": label,
}
df0 = pd.DataFrame(data=d)
print("finish reading data from c2db")
print("number of structures:", len(df0))


def main():
    mat_df_read = pd.read_excel(list_file)
    N = len(mat_df_read)
    count = 0
    original_id1 = []
    original_id2 = []
    mismatch = []
    elastic_energy = []
    for i in range(N):
        if divmod(i, 500)[1] == 0 and i > 1:
            print("complete", 500 * divmod(i, 500)[0], "steps")
        pair = mat_df_read.iloc[i]
        uid = pair["original_id"]
        uid1 = uid.split("'")[1]
        uid2 = uid.split("'")[3]
        mat1 = df0[df0["uid"] == uid1].iloc[0]
        mat2 = df0[df0["uid"] == uid2].iloc[0]
        subs = mat1["atoms"]
        film = mat2["atoms"]
        ela_energy_pre = pair["elastic_energy"]
        natom1 = subs.num_atoms
        natom2 = film.num_atoms
        try:
            hetero = make_interface(film=film, subs=subs, seperation=2.5, vacuum=20)
            hetatoms = hetero["interface"].num_atoms
            if ela_energy_pre < 0.01922:

                # make interface using equilibrium position under elasticity
                lat1sm = hetero["subs_sl"].lattice_mat
                lat2sm = hetero["film_sl"].lattice_mat
                C1 = heterostructure.elastic_tensor(mat1["elastic_tensor"])
                C2 = heterostructure.elastic_tensor(mat2["elastic_tensor"])
                (
                    ratio_min,
                    ela_energy,
                    Strain_u1,
                    Strain_u2,
                    Strain_v1,
                    Strain_v2,
                ) = Elastic.elastic(lat1sm, lat2sm, C1, C2)
                Strain_1 = [Strain_u1, Strain_u2, 0.0]
                Strain_2 = [Strain_v1, Strain_v2, 0.0]
                # compute area of the strained unit cell
                elastic_energy.append(ela_energy_pre)
                # strain the lattice and preserve fractional coordinates
                film_eq = film.strain_atoms(Strain_2)
                subs_eq = subs.strain_atoms(Strain_1)
                film_eq.frac_coords = film.frac_coords
                subs_eq.frac_coords = subs.frac_coords
                film_eq.cart_coords = film_eq.lattice.cart_coords(film_eq.frac_coords)
                subs_eq.cart_coords = subs_eq.lattice.cart_coords(subs_eq.frac_coords)
                original_id1.append(uid1)
                original_id2.append(uid2)
                mismatch.append([abs(hetero["mismatch_u"]), abs(hetero["mismatch_v"])])
                # Rot_Center computes translation vector (after putting center of mass on the origin)
                # to make atoms on corresponding maximal wyckoff positions with rotation center at the origin
                # apply flipping first
                R_film, R_subs = heterostructure.Rot_Center(film_eq, subs_eq, 0, 0)
                film_eq = film_eq.center_around_origin(R_film)
                subs_eq = subs_eq.center_around_origin(R_subs)
                film_flipped_list = heterostructure.Apply_flipping(film_eq)
                subs_flipped_list = heterostructure.Apply_flipping(subs_eq)
                heter_atom = []

                for film_temp_flip in film_flipped_list:
                    heter_atom_temp1 = []
                    for subs_temp_flip in subs_flipped_list:
                        film_eq_list = heterostructure.Apply_Rotation(film_temp_flip)
                        heter_atom_temp0 = []
                        for ind_film, film_temp in enumerate(film_eq_list):
                            # Compute the displacement of center of mass to make the alignment of corresponding maximal wyckoff positions
                            # DISPLACEMENT_wyckoff(film,subs,wyckoff_indices_film,wyckoff_indices_substrate) output displacement
                            # (fractional coordinates with respect to substrate lattice
                            # Displacement_list = DISPLACEMENT_wyckoff_list(film_temp,subs_temp_flip, 0, 0)
                            # compute translation
                            t_list = heterostructure.Translation_List(
                                film_temp, subs_temp_flip
                            )
                            # make heterostructure from translation

                            # for Displacement in Displacement_list:
                            #    heter_atom_temp0 = []
                            #    het_list_before_check = Make_Het_Translation(film_temp,subs_temp_flip,t_list, Displacement, seperation=3,vacuum=20)
                            Displacement = np.array([0.0, 0.0, 0.0])
                            het_list_before_check = heterostructure.Make_Het_Translation(
                                film_temp,
                                subs_temp_flip,
                                R_film,
                                R_subs,
                                t_list,
                                Displacement,
                                seperation=3,
                                vacuum=20,
                            )
                            het_list_before_check = heterostructure.exclude_duplicate(
                                het_list_before_check
                            )
                            for het_temp in het_list_before_check:
                                heter_atom_temp0.append(het_temp)
                        heter_atom_temp0 = heterostructure.exclude_duplicate(
                            heter_atom_temp0
                        )
                        for het in heter_atom_temp0:
                            heter_atom_temp1.append(het)
                    heter_atom_temp1 = heterostructure.exclude_duplicate(heter_atom_temp1)
                    for het in heter_atom_temp1:
                        heter_atom.append(het)

                # check redundency
                heter_atom1 = heterostructure.exclude_duplicate(heter_atom)
                # fine-selection of vdW gap
                heter_vdw = heterostructure.gap_adjust(heter_atom1, R_film, R_subs)
                # save heterostructure to Poscar file
                layers = [uid1.split("-")[0], uid2.split("-")[0]]
                ind_mono = [uid1.split("-")[1], uid2.split("-")[1]]
                count += len(heter_atom1)
                for j, het in enumerate(heter_vdw):
                    # if j % 5 == 2:
                    het_atom = het["interface"].center_around_origin()
                    # ind_2 = chr(97 + j % 5)
                    ind_1 = str(int(j))
                    Poscar(het_atom).write_file(
                        "./het_poscar/POSCAR_"
                        + layers[0]
                        + "_"
                        + ind_mono[0]
                        + "_"
                        + layers[1]
                        + "_"
                        + ind_mono[1]
                        + "_"
                        + ind_1
                        + ".vasp"
                    )
            # Poscar(subs).write_file('/Users/user/Documents/pycode/Lattice matching/data/c2db_monolayer_poscar/POSCAR_'+layers[0]+'-'+ind_mono[0]+'.vasp')
            # Poscar(film).write_file('/Users/user/Documents/pycode/Lattice matching/data/c2db_monolayer_poscar/POSCAR_'+layers[1]+'-'+ind_mono[1]+'.vasp')
    
        except Exception as error:
            print(error)

    d = {
        "original_id1": original_id1,
        "original_id2": original_id2,
        "mismatch": mismatch,
        "elastic_energy": elastic_energy,
    }
    mat_df_fine = pd.DataFrame(data=d)

    # Define the file path and name
    file_path = "./c2db_used_pairs.xlsx"
    # Output the DataFrame to an Excel file
    mat_df_fine.to_excel(file_path, index=False)

    print(f"DataFrame has been written to {file_path}")


main()
