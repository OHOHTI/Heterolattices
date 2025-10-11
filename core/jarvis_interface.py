'''
This module is used to generate heterostructures using the Zur algorithm. Modified from Jarvis-Tools.
https://github.com/usnistgov/jarvis

'''
from itertools import product
import numpy as np
from jarvis.core.atoms import fix_pbc
from jarvis.core.lattice import Lattice, lattice_coords_transformer
from jarvis.core.atoms import Atoms


# redefined add_atoms function, removing a part which transform top layer cartesian coordinates
# not use it because need to metain cartesian coordinates while changing fractional coordinates when lattice vectors change to ones of the bottom layer
def add_atoms(top, bottom, r_top, r_bottom, distance=[0, 0, 1], apply_strain=False):
    """
    Add top and bottom Atoms with a distance array.

    Bottom Atoms lattice-matrix is chosen as final lattice.
    """
    top = top.center_around_origin(r_top)
    bottom = bottom.center_around_origin(r_bottom + distance)
    strain_x = (top.lattice_mat[0][0] - bottom.lattice_mat[0][0]) / bottom.lattice_mat[
        0
    ][0]
    strain_y = (top.lattice_mat[1][1] - bottom.lattice_mat[1][1]) / bottom.lattice_mat[
        1
    ][1]
    if apply_strain:
        top.apply_strain([strain_x, strain_y, 0])
    #  print("strain_x,strain_y", strain_x, strain_y)
    elements = []
    coords = []
    lattice_mat = bottom.lattice_mat
    for i, j in zip(bottom.elements, bottom.frac_coords):
        elements.append(i)
        coords.append(j)

    # top_cart_coords = lattice_coords_transformer(
    # new_lattice_mat=top.lattice_mat,
    # old_lattice_mat=bottom.lattice_mat,
    # cart_coords=top.cart_coords,
    # )
    top_frac_coords = bottom.lattice.frac_coords(top.cart_coords)
    for i, j in zip(top.elements, top_frac_coords):
        elements.append(i)
        coords.append(j)

    order = np.argsort(np.array(elements))
    # elements = np.array(elements)[order]
    # coords = np.array(coords)[order]
    determnt = np.linalg.det(np.array(lattice_mat))
    if determnt < 0.0:
        lattice_mat = -1 * np.array(lattice_mat)
    determnt = np.linalg.det(np.array(lattice_mat))
    if determnt < 0.0:
        print("Serious issue, check lattice vectors.")
        print("Many software follow right hand basis rule only.")
    combined = Atoms(
        lattice_mat=lattice_mat,
        coords=coords,
        elements=elements,
        cartesian=False,
    ).center_around_origin()
    return combined


class ZSLGenerator(object):
    """
    Uses Zur algorithm to find best matched interfaces.

    This class is modified from pymatgen.
    """

    def __init__(
        self,
        max_area_ratio_tol=0.09,
        max_area=400,
        max_length_tol=0.03,
        max_angle_tol=0.01,
    ):
        """
        Intialize for a specific film and substrate.

        Parameters for the class.
        Args:
            max_area_ratio_tol(float): Max tolerance on ratio of
                super-lattices to consider equal

            max_area(float): max super lattice area to generate in search

            max_length_tol: maximum length tolerance in checking if two
                vectors are of nearly the same length

            max_angle_tol: maximum angle tolerance in checking of two sets
                of vectors have nearly the same angle between them
        """
        self.max_area_ratio_tol = max_area_ratio_tol
        self.max_area = max_area
        self.max_length_tol = max_length_tol
        self.max_angle_tol = max_angle_tol

    def is_same_vectors(self, vec_set1, vec_set2):
        """
        Check two sets of vectors are the same.

        Args:
            vec_set1(array[array]): an array of two vectors

            vec_set2(array[array]): second array of two vectors
        """
        if np.absolute(rel_strain(vec_set1[0], vec_set2[0])) > self.max_length_tol:
            return False
        elif np.absolute(rel_strain(vec_set1[1], vec_set2[1])) > self.max_length_tol:
            return False
        elif np.absolute(rel_angle(vec_set1, vec_set2)) > self.max_angle_tol:
            return False
        else:
            return True

    def generate_sl_transformation_sets(self, film_area, substrate_area):
        """Generate transformation sets for film/substrate.

        The transformation sets map the film and substrate unit cells to super
        lattices with a maximum area.

        Args:

            film_area(int): the unit cell area for the film.

            substrate_area(int): the unit cell area for the substrate.

        Returns:
            transformation_sets: a set of transformation_sets defined as:
                1.) the transformation matricies for the film to create a
                super lattice of area i*film area
                2.) the tranformation matricies for the substrate to create
                a super lattice of area j*film area
        """
        transformation_indicies = [
            (i, j)
            for i in range(1, int(self.max_area / film_area))
            for j in range(1, int(self.max_area / substrate_area))
            if np.absolute(film_area / substrate_area - float(j) / i)
            < self.max_area_ratio_tol
        ]

        # Sort sets by the square of the matching area and yield in order
        # from smallest to largest
        for x in sorted(transformation_indicies, key=lambda x: x[0] * x[1]):
            yield (
                gen_sl_transform_matricies(x[0]),
                gen_sl_transform_matricies(x[1]),
            )

    @staticmethod
    def same_length_order(u_set, v_set):
        """
        two sets of vectors u and v contains two 2d vectors
        reorder vectors in v_set so that they have same length as the vector in u_set in the same potisions
        """
        u_set = np.array(u_set)
        v_set = np.array(v_set)
        v_len = np.zeros(
            2,
        )
        same_len = 0
        for i in range(len(v_set)):
            v_len[i] = fast_norm(v_set[i])
        if np.abs(v_len[0] - v_len[1]) < 1e-2 * v_len[0]:
            same_len = 1
        if same_len != 1:
            u1 = u_set[0]
            i = np.argmin(np.abs(v_len - fast_norm(u1)))
            v1 = v_set[i]
            if i == 1:
                v2 = v_set[0]
            else:
                v2 = v_set[1]
            v_set_reorder = [v1, v2]
            return u_set, v_set_reorder
        else:
            return u_set, v_set

    def get_equiv_transformations(
        self, transformation_sets, film_vectors, substrate_vectors
    ):
        """
        Apply the transformation_sets to the film and substrate vectors.

        Generate super-lattices and checks if they matches.
        Returns all matching vectors sets.

        Args:
            transformation_sets(array): an array of transformation sets:
                each transformation set is an array with the (i,j)
                indicating the area multipes of the film and subtrate it
                corresponds to, an array with all possible transformations
                for the film area multiple i and another array for the
                substrate area multiple j.

            film_vectors(array): film vectors to generate super lattices.

            substrate_vectors(array): substrate vectors to generate super
                lattices
        """
        for (
            film_transformations,
            substrate_transformations,
        ) in transformation_sets:
            # Apply transformations and reduce using Zur reduce methodology
            films = [
                reduce_vectors(*np.dot(f, film_vectors)) for f in film_transformations
            ]

            substrates = [
                reduce_vectors(*np.dot(s, substrate_vectors))
                for s in substrate_transformations
            ]

            # Check if equivalant super lattices
            for (f_trans, s_trans), (f, s) in zip(
                product(film_transformations, substrate_transformations),
                product(films, substrates),
            ):
                if self.is_same_vectors(f, s):
                    # apply rotation to align vectors
                    v1 = [f[0][0], f[0][1]]
                    v2 = [f[1][0], f[1][1]]
                    u1 = [s[0][0], s[0][1]]
                    u2 = [s[1][0], s[1][1]]
                    u_set = [u1, u2]
                    v_set = [v1, v2]
                    u_set, v_set = self.same_length_order(u_set, v_set)
                    # apply rotation to align a vector at first
                    u_temp = u_set[0]
                    v_temp = v_set[0]
                    theta = vec_angle(v_temp, u_temp)
                    Mrot = np.array(
                        [
                            [np.cos(theta), -np.sin(theta)],
                            [np.sin(theta), np.cos(theta)],
                        ]
                    )
                    v1 = np.matmul(Mrot, v1)
                    v2 = np.matmul(Mrot, v2)
                    f = [[v1[0], v1[1], 0.0], [v2[0], v2[1], 0.0]]
                    yield [f, s, f_trans, s_trans]

    def __call__(self, film_vectors, substrate_vectors, lowest=False):
        """Run the ZSL algorithm to generate all possible matching."""
        film_area = vec_area(*film_vectors)
        substrate_area = vec_area(*substrate_vectors)

        # Generate all super lattice comnbinations for a given set of miller
        # indicies
        transformation_sets = self.generate_sl_transformation_sets(
            film_area, substrate_area
        )

        # Check each super-lattice pair to see if they match
        for match in self.get_equiv_transformations(
            transformation_sets, film_vectors, substrate_vectors
        ):
            # Yield the match area, the miller indicies,
            yield self.match_as_dict(
                match[0],
                match[1],
                film_vectors,
                substrate_vectors,
                vec_area(*match[0]),
                match[2],
                match[3],
            )

            if lowest:
                break

    def match_as_dict(
        self,
        film_sl_vectors,
        substrate_sl_vectors,
        film_vectors,
        substrate_vectors,
        match_area,
        film_transformation,
        substrate_transformation,
    ):
        """
        Return dict which contains ZSL match.

        Args:
            film_miller(array)

            substrate_miller(array)
        """
        d = {}
        d["film_sl_vecs"] = np.asarray(film_sl_vectors)
        d["sub_sl_vecs"] = np.asarray(substrate_sl_vectors)
        d["match_area"] = match_area
        d["film_vecs"] = np.asarray(film_vectors)
        d["sub_vecs"] = np.asarray(substrate_vectors)
        d["film_transformation"] = np.asarray(film_transformation)
        d["substrate_transformation"] = np.asarray(substrate_transformation)

        return d


def gen_sl_transform_matricies(area_multiple):
    """
    Generate the transformation matricies.

    Convert a set of 2D vectors into a super
    lattice of integer area multiple as proven
    in Cassels:
    Cassels, John William Scott. An introduction to the geometry of
    numbers. Springer Science & Business Media, 2012.

    Args:
        area_multiple(int): integer multiple of unit cell area for super
        lattice area.

    Returns:
        matrix_list: transformation matricies to covert unit vectors to
        super lattice vectors.
    """
    return [
        np.array(((i, j), (0, area_multiple / i)))
        for i in get_factors(area_multiple)
        for j in range(area_multiple // i)
    ]


def rel_strain(vec1, vec2):
    """Calculate relative strain between two vectors."""
    return fast_norm(vec2) / fast_norm(vec1) - 1


def rel_angle(vec_set1, vec_set2):
    """
    Calculate the relative angle between two vector sets.

    Args:
        vec_set1(array[array]): an array of two vectors.

        vec_set2(array[array]): second array of two vectors.
    """
    return vec_angle(vec_set2[0], vec_set2[1]) / vec_angle(vec_set1[0], vec_set1[1]) - 1


def fast_norm(a):
    """Much faster variant of numpy linalg norm."""
    return np.sqrt(np.dot(a, a))


def vec_angle(a, b):
    """Calculate angle between two vectors."""
    cosang = np.dot(a, b)
    sinang = fast_norm(np.cross(a, b))
    return np.arctan2(sinang, cosang)


def vec_area(a, b):
    """Area of lattice plane defined by two vectors."""
    return fast_norm(np.cross(a, b))


def reduce_vectors(a, b):
    """Generate independent and unique basis vectors based on Zur et al."""
    if np.dot(a, b) < 0:
        return reduce_vectors(a, -b)
    if fast_norm(a) > fast_norm(b):
        return reduce_vectors(b, a)
    if fast_norm(b) > fast_norm(np.add(b, a)):
        return reduce_vectors(a, np.add(b, a))
    if fast_norm(b) > fast_norm(np.subtract(b, a)):
        return reduce_vectors(a, np.subtract(b, a))
    return [a, b]


def get_factors(n):
    """Generate all factors of n."""
    for x in range(1, n + 1):
        if n % x == 0:
            yield x


def if_align(v1_new, v1):
    a = 0
    if np.linalg.norm(np.cross(v1_new, v1)) < 1e-1 * np.linalg.norm(
        v1
    ) * np.linalg.norm(v1_new):
        a = 1
    return a


def align(lat1, lat2):
    v1 = np.array([lat1[0][0], lat1[0][1]])
    v2 = np.array([lat1[1][0], lat1[1][1]])
    u1 = np.array([lat2[0][0], lat2[0][1]])
    u2 = np.array([lat2[1][0], lat2[1][1]])
    if np.dot(v1, v2) < 0:
        v3 = v1 + v2
        rot3 = [1, 1, 0]
    else:
        v3 = v1 - v2
        rot3 = [1, -1, 0]

    if if_align(v1, u1):
        v1_n = v1
        if np.dot(v1_n, u1) > 0:
            rot1 = [1, 0, 0]
        else:
            rot1 = [-1, 0, 0]
    elif if_align(v2, u1):
        v1_n = v2
        if np.dot(v1_n, u1) > 0:
            rot1 = [0, 1, 0]
        else:
            rot1 = [0, -1, 0]
    elif if_align(v3, u1):
        v1_n = v3
        if np.dot(v1_n, u1) > 0:
            rot1 = rot3
        else:
            rot1 = -1 * np.array(rot3)

    if if_align(v1, u2):
        v2_n = v1
        if np.dot(v2_n, u2) > 0:
            rot2 = [1, 0, 0]
        else:
            rot2 = [-1, 0, 0]
    elif if_align(v2, u2):
        v2_n = v2
        if np.dot(v2_n, u2) > 0:
            rot2 = [0, 1, 0]
        else:
            rot2 = [0, -1, 0]
    elif if_align(v3, u2):
        v2_n = v3
        if np.dot(v2_n, u2) > 0:
            rot2 = rot3
        else:
            rot2 = -1 * np.array(rot3)

    lat1_n = [[v1_n[0], v1_n[1], 0], [v2_n[0], v2_n[1], 0], lat1[2]]
    rot = np.array([rot1, rot2, [0, 0, 1]])
    return rot


def make_interface(
    film="",
    subs="",
    r_film=np.zeros(3),
    r_subs=np.zeros(3),
    atol=1,
    ltol=0.05,
    max_area=500,
    max_area_ratio_tol=1.00,
    seperation=4.0,
    vacuum=8.0,
    apply_strain=False,
    shift=0.0,
):
    """
    Use as main function for making interfaces/heterostructures.

    Return mismatch and other information as info dict.

    Args:
       film: top/film material.

       subs: substrate/bottom/fixed material.

       seperation: minimum seperation between two.

       vacuum: vacuum will be added on both sides.
       So 2*vacuum will be added.
    """
    z = ZSLGenerator(
        max_area_ratio_tol=max_area_ratio_tol,
        max_area=max_area,
        max_length_tol=ltol,
        max_angle_tol=atol,
    )
    film = fix_pbc(film.center_around_origin(r_film))
    subs = fix_pbc(subs.center_around_origin(r_subs))
    matches = list(z(film.lattice_mat[:2], subs.lattice_mat[:2], lowest=True))
    info = {}
    info["mismatch_u"] = "na"
    info["mismatch_v"] = "na"
    info["mismatch_angle"] = "na"
    info["area1"] = "na"
    info["area2"] = "na"
    info["film_sl"] = "na"
    info["matches"] = matches
    info["subs_sl"] = "na"
    uv1 = matches[0]["sub_sl_vecs"]
    uv2 = matches[0]["film_sl_vecs"]
    u = np.array(uv1)
    v = np.array(uv2)
    a1 = u[0]
    a2 = u[1]
    b1 = v[0]
    b2 = v[1]
    mismatch_u = np.linalg.norm(b1) / np.linalg.norm(a1) - 1
    mismatch_v = np.linalg.norm(b2) / np.linalg.norm(a2) - 1
    angle1 = (
        np.arccos(np.dot(a1, a2) / np.linalg.norm(a1) / np.linalg.norm(a2))
        * 180
        / np.pi
    )
    angle2 = (
        np.arccos(np.dot(b1, b2) / np.linalg.norm(b1) / np.linalg.norm(b2))
        * 180
        / np.pi
    )
    mismatch_angle = abs(angle1 - angle2)
    area1 = np.linalg.norm(np.cross(a1, a2))
    area2 = np.linalg.norm(np.cross(b1, b2))
    uv_substrate = uv1
    uv_film = uv2
    substrate_latt = Lattice(
        np.array([uv_substrate[0][:], uv_substrate[1][:], subs.lattice_mat[2, :]])
    )
    _, __, scell = subs.lattice.find_matches(substrate_latt, ltol=ltol, atol=atol)
    film_latt = Lattice(
        np.array([uv_film[0][:], uv_film[1][:], film.lattice_mat[2, :]])
    )
    scell[2] = np.array([0, 0, 1])
    scell_subs = scell
    _, __, scell = film.lattice.find_matches(film_latt, ltol=ltol, atol=atol)
    scell[2] = np.array([0, 0, 1])
    scell_film = scell

    film_temp = film.make_supercell_matrix(scell_film)
    subs_temp = subs.make_supercell_matrix(scell_subs)

    # align the superlattice vectors with the original lattice vectors
    film_sc_mat = align(film_temp.lattice.matrix, film.lattice.matrix)
    subs_sc_mat = align(subs_temp.lattice.matrix, subs.lattice.matrix)

    film_scell = film_temp.make_supercell_matrix(film_sc_mat)
    subs_scell = subs_temp.make_supercell_matrix(subs_sc_mat)
    info["mismatch_u"] = mismatch_u
    info["mismatch_v"] = mismatch_v
    info["mismatch_angle"] = mismatch_angle
    info["area1"] = area1
    info["area2"] = area2
    info["film_sl"] = film_scell
    info["subs_sl"] = subs_scell
    info["shift"] = shift
    info["seperation"] = seperation
    substrate_top_z = max(np.array(subs_scell.cart_coords)[:, 2])
    substrate_bot_z = min(np.array(subs_scell.cart_coords)[:, 2])
    film_top_z = max(np.array(film_scell.cart_coords)[:, 2])
    film_bottom_z = min(np.array(film_scell.cart_coords)[:, 2])
    thickness_sub = abs(substrate_top_z - substrate_bot_z)
    thickness_film = abs(film_top_z - film_bottom_z)
    sub_z = (
        (vacuum + substrate_top_z)
        * np.array(subs_scell.lattice_mat[2, :])
        / np.linalg.norm(subs_scell.lattice_mat[2, :])
    )
    shift_normal = (
        -sub_z
        / np.linalg.norm(sub_z)
        * seperation
        / np.linalg.norm(subs_scell.lattice_mat[2, :])
    )
    shift_in_plane = shift
    # tmp = (
    # thickness_film / 2 + seperation + thickness_sub / 2
    # ) / np.linalg.norm(subs_scell.lattice_mat[2, :])
    # shift_normal = (
    # tmp
    # * np.array(subs_scell.lattice_mat[2, :])
    # / np.linalg.norm(subs_scell.lattice_mat[2, :])
    # )
    interface = add_atoms(
        film_scell,
        subs_scell,
        r_film,
        r_subs,
        shift_normal + shift_in_plane,
        apply_strain=apply_strain,
    ).center_around_origin([0, 0, 0])
    combined = interface.center(vacuum=vacuum).center_around_origin([0, 0, 0])
    info["interface"] = combined
    return info
