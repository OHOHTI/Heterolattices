import numpy as np

#Strain tensor calculation is based on the paper Acta Cryst. (1978). A34, 52-54

def strain_tensor(u1,u2,v1,v2):#original lattice vectors u1, u2, strained lattice vectors v1, v2
    #compute S matrix
    #as long as u and v are in the same unit, one does not need to worry about unit in later calculatoin
    u1_2d = [u1[0],u1[1]]
    u2_2d = [u2[0],u2[1]]
    Su = np.array([u1_2d,u2_2d])

    v1_2d = [v1[0],v1[1]]
    v2_2d = [v2[0],v2[1]]
    Sv = np.array([v1_2d,v2_2d])
    #compute strain tensor (linear Lagrangian strain tensor)
    str = 1/2 * (np.transpose(np.matmul(np.linalg.inv(Su),Sv)) + np.matmul(np.linalg.inv(Su),Sv)) - np.identity(2)
    return str


def el_Eng(str,c):
    e11 = str[0,0]
    e22 = str[1,1]
    e12 = str[0,1]
    e_vec = np.array([e11,e22,0,0,0,e12])
    c_arr = np.array(c)
    el_eng = 1/2 * np.dot(e_vec,np.matmul(c_arr,e_vec))
    return np.absolute(el_eng)

def if_align(v1_new,v1):
    a = 0
    if np.linalg.norm(np.cross(v1_new,v1)) < 1e-1 * np.linalg.norm(v1) * np.linalg.norm(v1_new):
        a = 1
    return a

#align two lattice vectors with another two
def align(v1,v2,u1,u2):
    if np.dot(v1,v2) < 0:
        v3 = v1 + v2
    else:
        v3 = v1 - v2

    if if_align(v1,u1):
        v1_n = v1
    elif if_align(v2,u1):
        v1_n = v2
    elif if_align(v3,u1):
        v1_n = v3
    if if_align(v1,u2):
        v2_n = v1
    elif if_align(v2,u2):
        v2_n = v2
    elif if_align(v3,u2):
        v2_n = v3
        
    if np.dot(v1_n,u1) > 0:
        v1 = v1_n
    else:
        v1 = - v1_n

    if np.dot(v2_n,u2) > 0:
        v2 = v2_n
    else:
        v2 = - v2_n

    return v1_n,v2_n

def fast_norm(a):
    """Much faster variant of numpy linalg norm."""
    return np.sqrt(np.dot(a, a))

def vec_angle(a, b):
    """Calculate angle between two vectors."""
    cosang = np.dot(a, b)
    sinang = fast_norm(np.cross(a, b))
    return np.arctan2(sinang, cosang)
    
def same_length_order(u_set, v_set):
        """
        two sets of vectors u and v contains two 2d vectors
        reorder vectors in v_set so that they have same length as the vector in u_set in the same potisions
        """
        u_set = np.array(u_set)
        v_set = np.array(v_set)
        v_len = np.zeros(2,)
        same_len = 0
        for i in range(len(v_set)):
            v_len[i] = fast_norm(v_set[i])
        if np.abs(v_len[0] -  v_len[1]) < 1e-2 * v_len[0]:
            same_len = 1
        if same_len != 1:
            u1 = u_set[0]
            i = np.argmin(np.abs(v_len - fast_norm(u1)))
            v1 = v_set[i]
            if i == 1:
                v2 = v_set[0]
            else:
                v2 = v_set[1]
            v_set_reorder = [v1,v2]
            return u_set, v_set_reorder
        else:
            return u_set, v_set

def elastic(lat1sm,lat2sm,C1,C2):
    """
    lat1sm, C1: lattice and elastic tensor of substrate
    lat2sm, C2: lattice and elastic tensor of film
    """
    #align the superlattice vectors
    u1 = np.array([lat1sm[0][0],lat1sm[0][1]])
    u2 = np.array([lat1sm[1][0],lat1sm[1][1]])
    v1 = np.array([lat2sm[0][0],lat2sm[0][1]])
    v2 = np.array([lat2sm[1][0],lat2sm[1][1]])

    #apply rotation to align one of the vectors
    u_set = [u1,u2]
    v_set = [v1,v2]
    u_set, v_set = same_length_order(u_set,v_set)
    #align the first lattice vectors of film and substrate
    u_temp = u_set[0]
    v_temp = v_set[0]
    theta = vec_angle(v_temp, u_temp)
    Mrot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    v1 = np.matmul(Mrot, v1)
    v2 = np.matmul(Mrot, v2)
    #find correspondence lattice vectors of film with respect to those of the substrate
    if np.dot(v1,v2) < 0:
        v3 = v1 + v2
    else:
        v3 = v1 - v2

    if if_align(v1,u1):
        v1_n = v1
    elif if_align(v2,u1):
        v1_n = v2
    elif if_align(v3,u1):
        v1_n = v3
    if if_align(v1,u2):
        v2_n = v1
    elif if_align(v2,u2):
        v2_n = v2
    elif if_align(v3,u2):
        v2_n = v3

    if np.dot(v1_n,u1) > 0:
        v1 = v1_n
    else:
        v1 = - v1_n

    if np.dot(v2_n,u2) > 0:
        v2 = v2_n
    else:
        v2 = - v2_n

    #compute equivalent point
    df1 = v1 - u1
    df2 = v2 - u2
    nstep = 20
    el_list = []
    for i in range(nstep + 1):
        ratio = i/nstep
        #deformation vectors
        df1_uv = df1 * ratio
        df2_uv = df2 * ratio
        df1_vu = - (df1 - df1_uv)
        df2_vu = - (df2 - df2_uv)
        #strained lattice vectors. ratio = 0: u -> v, v -> v, ratio = 1: u -> u, v -> u
        us1 = df1_uv + u1
        us2 = df2_uv + u2
        vs1 = df1_vu + v1
        vs2 = df2_vu + v2
        stra1 = strain_tensor(u1,u2,us1,us2)
        stra2 = strain_tensor(v1,v2,vs1,vs2)
        el_eng1 = el_Eng(stra1,C1)
        el_eng2 = el_Eng(stra2,C2)
        el_list.append(el_eng1 + el_eng2)
    min_ind = el_list.index(min(el_list))
    ratio_min = min_ind/nstep
    ela_energy = min(el_list)
    df1_uv = df1 * ratio_min
    df2_uv = df2 * ratio_min
    #strained lattice vectors. ratio = 0: u -> v, v -> v, ratio = 1: u -> u, v -> u
    uf1 = df1_uv + u1
    uf2 = df2_uv + u2
    strain_u1 = np.sign(np.linalg.norm(uf1) - np.linalg.norm(u1)) * np.linalg.norm(df1_uv)/np.linalg.norm(u1)
    strain_u2 = np.sign(np.linalg.norm(uf2) - np.linalg.norm(u2)) * np.linalg.norm(df2_uv)/np.linalg.norm(u2)
    hetlatsm1 = [[uf1[0],uf1[1],0],[uf2[0],uf2[1],0],lat1sm[2]]
    #align back vf
    v1o = np.array([lat2sm[0][0],lat2sm[0][1]])
    v2o = np.array([lat2sm[1][0],lat2sm[1][1]])
    if np.dot(uf1,uf2) < 0:
        uf3 = uf1 + uf2
    else:
        uf3 = uf1 - uf2
    if if_align(uf1,v1o):
        vf1 = uf1
    elif if_align(uf2,v1o):
        vf1 = uf2
    elif if_align(uf3,v1o):
        vf1= uf3
    if if_align(uf1,v2o):
        vf2 = uf1
    elif if_align(uf2,v2o):
        vf2 = uf2
    elif if_align(uf3,v2o):
        vf2 = uf3

    if np.dot(vf1,v1o) > 0:
        vf1 = vf1
    else:
        vf1 = - vf1

    if np.dot(vf2,v2o) > 0:
        vf2 = vf2
    else:
        vf2 = - vf2
    hetlatsm2 = [[vf1[0],vf1[1],0],[vf2[0],vf2[1],0],lat1sm[2]]
    #strain of v vectors are computed with respect to the original (non-aligned) lattice vectors
    strain_v1 = np.sign(np.linalg.norm(vf1) - np.linalg.norm(v1o)) * np.linalg.norm(vf1 - v1o)/np.linalg.norm(v1o)
    strain_v2 = np.sign(np.linalg.norm(vf2) - np.linalg.norm(v2o)) * np.linalg.norm(vf2 - v2o)/np.linalg.norm(v2o)

    return ratio_min, ela_energy, strain_u1, strain_u2, strain_v1, strain_v2 
