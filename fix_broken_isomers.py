#! /bin/env python3
import sys
import logging
import argparse

import numpy as np
import MDAnalysis as mda


##################### DEFINITIONS              ###########################

def rotmax(vin, vax, ang):
    v1, v2, v3 = vax
    print("ax", vax)
    print("in", vin)
    cos = np.cos
    sin = np.sin
    dcos = lambda ang: (1-cos(ang))
    rotmat = np.matrix([
        [ ( v1**2 * dcos(ang) + cos(ang) ),    ( v1*v2*dcos(ang) - v3*sin(ang) ), ( v1*v3*dcos(ang) + v2*sin(ang) ) ],
        [ ( v1*v2*dcos(ang) + v3*sin(ang) ), ( v2**2 * dcos(ang) + cos(ang)  ), ( v2*v3*dcos(ang) - v1*sin(ang) ) ],
        [ (v3*v1*dcos(ang) - v2*sin(ang) ),  ( v3*v2*dcos(ang) + v1*sin(ang) ), ( v3**2*dcos(ang) + cos(ang) ) ],
    ])
    print("rot", rotmat)
    vout = np.multiply(rotmat, vin.transpose())
    print("out", vout)
    return vout

def switch_chiral_methyl(Hatom, CMe, H3Me):
    ''' switches C and H position and places H atoms relative around new positon'''
    h_pos = Hatom.position
    c_pos = CMe.position

    #cc = c_pos - np.mean([c_pos - i.position for i  in H3Me])
    #print(cc, )
    #rotax = np.cross(cc, c_pos-H3Me[0].position)
    #for hatm in H3Me:
    #    ch = c_pos - hatm.position
    #hatm_pos = [ rotmax(hatm.position-c_pos, rotax, (np.pi/180)*109)+h_pos for hatm in H3Me ]
    hatm_pos = [ (hatm.position-c_pos)*0.3 + h_pos for hatm in H3Me ]
    Hatom.position = c_pos
    CMe.position = h_pos
    for i, hatm in enumerate(H3Me):
        hatm.position = hatm_pos[i]

#def switch_trans_atoms_PL(Ca, C1, C2, C3, C4, Ce, C1Hs, C2Hs):
#
#    # calculate distance to center of mass for C2 and C3 (double bond)
#    #COG = mda.AtomGroup([C1, C2, C3, C4]).center_of_geometry()
#    #middleCs = [C2,C3]
#    #COG_vecs = [ C2.position-COG, C3.position-COG ]
#    #COG_distances = [ np.linalg.norm(i) for i in COG_vecs ]
#
#    #ndx_mindist = np.argmin(COG_distances)
#    #if ndx_mindist == 1:
#    #    ndx_maxdist = 0
#    #else:
#    #    ndx_maxdist = 1
#
#    #LOGGER.debug("COG is %s, vecs %s", COG, COG_vecs)
#    #LOGGER.debug("old pos %s %s", C2.position, C3.position)
#
#    ## move atom that is closer to COG outside
#    #middleCs[ndx_mindist].position = middleCs[ndx_mindist].position - 10*COG_vecs[ndx_maxdist]
#    ## and move second atom slightly outside
#    #middleCs[ndx_maxdist].position = middleCs[ndx_maxdist].position - 10*COG_vecs[ndx_mindist]
#
#    #LOGGER.debug("new pos %s %s", C2.position, C3.position)
#
#    #4. flip Hs
#    for h in C1Hs:
#        LOGGER.debug("old pos H %s", h.position)
#        h.position = C1.position + (h.position - C1.position)
#        LOGGER.debug("new pos H %s", h.position)
#    #for h in C2Hs:
#    #    h.position = C2.position + (C2.position - h.position)

def switch_trans_atoms_PL(Cdoubl, Hdoubl, CH2, H2):
    CH = Cdoubl.position - Hdoubl.position
    H2mean = np.mean([CH2.position-i.position for i in H2])
    CH_shifted = CH - CH2.position

    ### Move CH2
    ### Flip adjacent H2 if not facing correctly ###
    cosang = np.dot(CH_shifted, H2mean) * np.linalg.norm(CH_shifted) * np.linalg.norm(H2mean)
    if cosang > 0.5: # if |dihedral| < 60°
        CH2.position = CH2.position - H2mean
        for h in H2:
            Ch = CH2.position - h.position
            h.position = CH2.position + Ch
    else:
        CH2.position = CH2.position + H2mean



    ### shift atoms in H-C=C-H bond ###
    new_Cdoubl = Cdoubl.position + CH
    new_Hdoubl = new_Cdoubl + CH
    Cdoubl.position = new_Cdoubl
    Hdoubl.position = new_Hdoubl



def switch_trans_atoms_ERG(C23, C24, H23, H24):
    C2324 = C23.position - C24.position
    CH23 = H23.position - C23.position
    CH24 = H24.position - C24.position

    new_pos_c23 = C23.position - ( -CH23  * (np.dot(-CH23, C2324)) )
    new_pos_c24 = C24.position - ( -CH24  * (np.dot(-CH24, -C2324))  )
    new_pos_h23 = new_pos_c23 - CH23
    new_pos_h24 = new_pos_c24 - CH24

    C23.position = new_pos_c23
    C24.position = new_pos_c24
    H23.position = new_pos_h23
    H24.position = new_pos_h24




#####################         LOGGER            #########################

LOGGER = logging.getLogger("fix_isomers")
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

fh = logging.FileHandler("fix_isomers.log")
fh.setLevel("DEBUG")
fh.setFormatter(formatter)



######################### ARGPARSE ARGUMENTS ############################

PARSER = argparse.ArgumentParser()

# Non optional parameters
PARSER.add_argument('-f', action="store", metavar='struct.gro', required=True,
                    help="input structue")
PARSER.add_argument('-s', action="store", metavar='wrong_conf.log', required=True,
                    help="list of wrong configuration")
PARSER.add_argument('-m', action="store", metavar='bondtype', required=True,
                    help="bondtype to fix")
# optional arguments
PARSER.add_argument('-o', action="store", metavar='output.gro', nargs='?', required=False,
                    help="output file name", default="structure_fixed.gro")
# On/off flags
PARSER.add_argument('--debug', action="store_true",
                    help="")


ARGS = PARSER.parse_args()

INP_F    = ARGS.f
INP_O    = ARGS.o
INP_S    = ARGS.s
INP_M    = ARGS.m

if ARGS.debug:
    LOGGER.addHandler(fh)
    LOGGER.setLevel("DEBUG")

##############################################################################


def main():

    u = mda.Universe(INP_F)
    with open(INP_S, "r") as f:
        for line in f:

            if "#" in line:
                line, comment = line.split("#")[0], "#" + ''.join(line.split("#")[1:])
            else:
                comment = ""

            if not line.replace("\n", "") or "bondtype" in line or ".map" in line: #header or empty
                continue
            cols = line.split()
            bondtype = cols[0]
            resid   = cols[1]
            atomnames = cols[2:]

            if bondtype != INP_M:
                continue

            print("at", line)

            residue = u.atoms.select_atoms("resid {}".format(resid)).residues[0]
            print("residue:", residue)

            atomname_str = ' '.join(atomnames)
            # Chiral of ERG -- switch Me and H position
            if atomname_str == "H20 C20 C17 C22 C21" or atomname_str == "H24 C24 C23 C25 C28":
                print(atomnames)
                if atomnames[0] == "H20":
                    CMe = residue.atoms.select_atoms("name {}".format("C21"))[0]
                    Hatm = residue.atoms.select_atoms("name {}".format("H20"))[0]
                    H3Me = residue.atoms.select_atoms("name {}".format("H21A H21B H21C"))
                elif atomnames[0] == "H24":
                    CMe = residue.atoms.select_atoms("name {}".format("C28"))[0]
                    Hatm = residue.atoms.select_atoms("name {}".format("H24"))[0]
                    H3Me = residue.atoms.select_atoms("name {}".format("H28A H28B H28C"))
                switch_chiral_methyl(Hatm, CMe, H3Me)

            elif atomname_str == "H17 C17 C13 C20 C16":
                C = residue.atoms.select_atoms("name C17")[0]
                H = residue.atoms.select_atoms("name H17")[0]
                CH = C.position - H.position
                C.position = C.position + CH * 0.3
                H.position = C.position + CH


            # incorrect cis bond of erg
            elif atomname_str == "C20 C22 C23 C24":
                #C23, C24, H23, H24
                C23 = residue.atoms.select_atoms("name C23")[0]
                C24 = residue.atoms.select_atoms("name C24")[0]
                H23 = residue.atoms.select_atoms("name H23")[0]
                H24 = residue.atoms.select_atoms("name H24")[0]
                print(C23, C24, H23, H24)
                switch_trans_atoms_ERG(C23, C24, H23, H24)

            # incorrect trans bond of pl
            elif atomname_str == "C28 C29 C210 C211" or atomname_str == "C38 C39 C310 C311":
                #Ca, C1, C2, C3, C4, Ce, C1Hs, C2Hs

                #C1, C2, C3, C4 = atomnames

                #Ca = C1[0] + str(int(C1[1:])-1)
                #Ce = C4[0] + str(int(C4[1:])+1)

                #chain_ndx = int(C1[1]) - 2
                #letters = [["X", "Y"], ["R", "S"]]
                #C1Hs = ["H{}{}".format(int(C1[2:]), i) for i in letters[chain_ndx]]
                #C4Hs = ["H{}{}".format(int(C4[2:]), i) for i in letters[chain_ndx]]

                #carbons = [Ca, C1, C2, C3, C4, Ce]
                #carbons = [residue.atoms.select_atoms("name {}".format(i))[0] for i in carbons]
                #hydr1 = [residue.atoms.select_atoms("name {}".format(i))[0] for i in C1Hs]
                #hydr2 = [residue.atoms.select_atoms("name {}".format(i))[0] for i in C4Hs]

                #switch_trans_atoms_PL(*carbons, hydr1, hydr2 )


                C1, C2, C3, C4 = atomnames
                chain_ndx = int(C1[1]) - 2
                letters = [["R", "S"], ["X", "Y"]]

                print("name H{}{}".format(C2[2:], letters[chain_ndx][0] ))

                Cdoubl = residue.atoms.select_atoms("name {}".format(C2))[0]
                Hdoubl = residue.atoms.select_atoms("name H{}{}".format(C2[2:], letters[chain_ndx][0] ))[0]
                CH2    = residue.atoms.select_atoms("name {}".format(C1))[0]

                H2 = ["H{}{}".format(int(C1[2:]), i) for i in letters[chain_ndx]]
                H2 = [residue.atoms.select_atoms("name {}".format(i))[0] for i in H2]

                switch_trans_atoms_PL(Cdoubl, Hdoubl, CH2, H2)


    u.atoms.write(INP_O)

if __name__ == "__main__":
    main()