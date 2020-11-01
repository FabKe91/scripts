import logging
import argparse
import math
import random
import numpy as np

import MDAnalysis as mda

#####################         LOGGER            #########################

LOGGER = logging.getLogger("calc_s")
LOGGER.setLevel("INFO")

# create console handler and set level to debug
ch = logging.StreamHandler()
ch.setLevel("INFO")
# create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# add formatter to ch
ch.setFormatter(formatter)
# add ch to logger
LOGGER.addHandler(ch)

fh = logging.FileHandler("debug.log")
fh.setLevel("DEBUG")
fh.setFormatter(formatter)

######################### ARGPARSE ARGUMENTS ############################

PARSER = argparse.ArgumentParser()

# Non optional parameters
PARSER.add_argument('-f', action="store", metavar='traj.trr', required=True,
                    help="trajectory")
PARSER.add_argument('-s', action="store", metavar='grofile.gro', required=True,
                    help="structure")
# optional arguments
PARSER.add_argument('-o', action="store", metavar='outfile', nargs='?', required=False,
                    help="output file name", default="")
PARSER.add_argument('-m', action="store", metavar='', nargs='?', required=False,
                    help="lipid", default="DPPC")
# On/off flags
PARSER.add_argument('--debug', action="store_true",
                    help="")


ARGS = PARSER.parse_args()

TRJ   = ARGS.f
OUTPUTFILENAME    = ARGS.o
GRO    = ARGS.s
LIPID = ARGS.m

################################### DEFINITIONS ##########################################
TAILCARBONS = {
    'DP':[['C22', 'C23', 'C24', 'C25', 'C26', 'C27', 'C28', 'C29', 'C210', 'C211', 'C212', 'C213', 'C214', 'C215', 'C216'],                  #16:0
          ['C32', 'C33', 'C34', 'C35', 'C36', 'C37', 'C38', 'C39', 'C310', 'C311', 'C312', 'C313', 'C314', 'C315', 'C316']],                 #16:0
    'DU':[['C22', 'C23', 'C24', 'C25', 'C26', 'C27', 'C28', 'C29', 'C210', 'C211', 'C212', 'C213', 'C214', 'C215', 'C216', 'C217', 'C218'],  #18:2
          ['C32', 'C33', 'C34', 'C35', 'C36', 'C37', 'C38', 'C39', 'C310', 'C311', 'C312', 'C313', 'C314', 'C315', 'C316', 'C317', 'C318']], #18:2
}

SCD_TAIL_ATOMS_OF = {
    'DP':[TAILCARBONS['DP'][0][::2], TAILCARBONS['DP'][1][::2]],
    'DU':[['C22', 'C24', 'C26', 'C28', 'C211', 'C214', 'C216', 'C218'], # Double bonds between 9-10, 12-12
            ['C32', 'C34', 'C36', 'C38', 'C311', 'C314', 'C316', 'C318']],
    }

def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    from: https://stackoverflow.com/questions/6802577/rotation-of-3d-vector
    """
    axis = np.asarray(axis)
    axis = axis / math.sqrt(np.dot(axis, axis))
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])

def get_rand_axis(lipvec):
    ''' '''
    xaxis = np.array([1,0,0])
    zaxis = np.array([0,0,1])
    gammarange = np.arange(-1, 1+0.01, 0.01)
    gammaweights = None
    thetarange = np.arange(0, 1+0.01, 0.01)
    thetaweights = [166, 766, 1102, 817, 962, 1018, 971, 1074, 1046, 1074, 1084, 1171, 1264, 1358, 1341, 1373, 1457, 1523, 1564, 1727, 1763, 1807, 1962, 1978, 2103, 2298, 2370, 2500, 2696, 2848, 3035, 3121, 3261, 3482, 3636, 3925, 4092, 4269, 4510, 4715, 5004, 5235, 5501, 5834, 5946, 6288, 6688, 6866, 7289, 7621, 8082, 8262, 8940, 9265, 9452, 9978, 10250, 10933, 11148, 11663, 12201, 12700, 13127, 13848, 14205, 14746, 15441, 16102, 16507, 17178, 17810, 18083, 18868, 19506, 20259, 20744, 21318, 22106, 22599, 23679, 23937, 25042, 25159, 26299, 26746, 27373, 27962, 28528, 28980, 29286, 30208, 30552, 30954, 31316, 31871, 32218, 32283, 33081, 33232, 33601, 33767]
    cosgamma = random.choices(gammarange, weights=gammaweights)[0]
    costheta = random.choices(thetarange, weights=thetaweights)[0]
    theta = np.arccos(costheta)
    gamma = np.arccos(cosgamma)
    #print("costheta, theta", costheta, theta)
    #print("gamma theta", gamma, theta)
    vec = np.dot(rotation_matrix(xaxis, theta), lipvec)  # rotation around x (z tilt)
    #print(vec)
    vec = np.dot(rotation_matrix(zaxis, gamma), vec) # rotation around z (xy plane)
    #print(vec)
    return vec



def get_cc_order(positions: [np.array,], ref_axis=(0,0,1)) -> float:
    ''' Calculate the cc order parameter

        cc order parameter is defined as 0.5 * ( ( 3 * (cos_angle**2)) - 1 )
        for each consecutive positions in the positions array, whole the cos_angle is
        calculated with respect to ref_axis

        Input positions must be a list of arrays of positions:
        The list holds the arrays containing atom positions
        while the first array holds positions of chain 1, the second of chain 2 ...
        e.g.: np.array( np.array(pos_sn1), np.array(pos_sn2) )


    '''
    assert isinstance(positions, list)

    s_vals = [[] for _ in positions]

    for sn_x, positions_sn_x in enumerate(positions):

        #scds_of_atoms = []
        for i in range( len(positions_sn_x) - 1 ):# Explicitly using range(len()) to save if clause

            pos1, pos2 = positions_sn_x[i], positions_sn_x[i+1]

            diffvector = pos2 - pos1
            diffvector /= np.linalg.norm(diffvector)

            cos_angle = np.dot(diffvector, ref_axis)
            #scds_of_atoms.append( 0.5 * ( ( 3 * (cos_angle**2)) - 1 )  )
            s_vals[sn_x].append( 0.5 * ( ( 3 * (cos_angle**2)) - 1 )  )

            LOGGER.debug("Diffvector %s", diffvector)
            LOGGER.debug("Resulting cos %s", cos_angle)
        s_vals[sn_x] = np.array( s_vals[sn_x] )

    return np.array(s_vals).mean(), s_vals

def create_cc_orderfiles():
    '''

    '''
    u = mda.Universe(GRO, TRJ)

    ## Gather all input data for _calc_scd_output function
    len_traj = len(u.trajectory)

    with open(OUTPUTFILENAME, "w") as scdfile:

        #### Print header files ####
        print("{: <12}{: <10}{: <15}"\
                .format("time", "axndx", "S"),
            file=scdfile)

        tailatms = SCD_TAIL_ATOMS_OF[LIPID[:2]]
        s_atoms = []
        for sn in tailatms:
            atms = u.atoms.select_atoms( "name {}"\
                    .format(' '.join(sn) ) )
            idmap = {(id,pos) for pos,id in enumerate(sn)}
            atms = sorted(atms, key=lambda atom:idmap[atom.name])
            s_atoms.append(atms)
        glycatms = mda.AtomGroup([u.atoms.select_atoms("name P"), u.atoms.select_atoms("name C1")])
        for t in range(len_traj):

            time = u.trajectory[t].time
            LOGGER.info("At time %s", time)

            tailatms = SCD_TAIL_ATOMS_OF[LIPID[:2]]
            positions = []
            for atms in s_atoms:
                positions.append([atm.position for atm in atms])

            glycvec = glycatms.positions[0] - glycatms.positions[1]
            for i in range(100):
                refaxis = get_rand_axis(glycvec)

                order_val, s_prof = get_cc_order(positions, ref_axis=refaxis)

                LOGGER.debug("printing to files ...")
                ### Print everything to files ###
                line_scd = "{: <12.2f}{: <10}{: <15.8}".format(
                        time, i, order_val)
                print(line_scd, file=scdfile)