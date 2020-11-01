#!/bin/env python3
import sys
import logging
import argparse

import numpy as np
import MDAnalysis as mda
from MDAnalysis.analysis.distances import distance_array

#####################       DEFINITIONS         #########################
HEADATM  = "P"
TAILATMS = ["C216", "C316"]


def chunks(lst, n):
    """Yield successive n-sized chunks from lst.
       From https://www.codegrepper.com/code-examples/typescript/splitting+a+list+into+pairs+python
    """
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


#####################         LOGGER            #########################

LOGGER = logging.getLogger("PL_head_tail_dist")
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

fh = logging.FileHandler("PL_head_tail_dist_debug.log")
fh.setLevel("DEBUG")
fh.setFormatter(formatter)



######################### ARGPARSE ARGUMENTS ############################

PARSER = argparse.ArgumentParser()

PARSER.add_argument('-f', action="store", metavar='traj.xtc', required=False, nargs="?",
                    help="trajectory")
PARSER.add_argument('-s', action="store", metavar='structure.gro', required=True,
                    help="structure file")
PARSER.add_argument('-l', action="store", metavar='lipidname', required=False, nargs="?",
                    help="lipid name", default="DPPC")
PARSER.add_argument('-o', action="store", metavar='head_tail_dist.dat', nargs='?', required=False,
                    help="output file name", default="head_tail_dist.dat")
PARSER.add_argument('-m', action="store", metavar='mode', required=True,
                    help="mode to calculate: distance or angle")
# On/off flags
PARSER.add_argument('--debug', action="store_true",
                    help="Logs all debug information to check_structure_debug.log")


ARGS = PARSER.parse_args()

STRUCTFNAME    = ARGS.s
OUTPUTFILENAME = ARGS.o
TRAJFILE       = ARGS.f
LIPIDNAME      = ARGS.l
MODE = ARGS.m

OUTPUTFILENAME = OUTPUTFILENAME.replace(".dat", MODE+".dat")

if ARGS.debug:
    LOGGER.addHandler(fh)
    LOGGER.setLevel("DEBUG")

def calc_distance():
    u = mda.Universe(STRUCTFNAME, TRAJFILE)
    head_atms = u.atoms.select_atoms("name {}".format(HEADATM))
    tail_atms = u.atoms.select_atoms("name {}".format(' '.join(TAILATMS)))
    tail_atms = mda.AtomGroup(list(chunks(tail_atms, len(TAILATMS))))

    LOGGER.debug("head_atms: %s tail atms %s", head_atms, tail_atms)

    if not isinstance(mda.AtomGroup, list):
        head_atms = mda.AtomGroup(head_atms)
    
    with open(OUTPUTFILENAME, "w") as outf:
        outf.write("{: <15}{: <10}{: <10}{: <20}\n".format("time", "resid", "chain", "dist"))
        for ts in u.trajectory:
            LOGGER.info("at time %s", ts.time)
            outp_inf = []
            #distances = distance_array(np.array([head.position for head in head_atms]), np.array([tail.position for tail in tail_atms]))
            for head, tail in zip(head_atms, tail_atms):
                #print("HEAD", head)
                distances = distance_array(head.position, tail.position)[0]
                
                #LOGGER.debug("distances:\n%s", distances)
                residue = head.residue
                #LOGGER.info("at residue %s", residue)
                for chainid, dist in enumerate(distances):
                    outpline = "{: <15}{: <10}{: <10}{: <20}\n"\
                        .format(ts.time, residue.resid, chainid, dist )
                    outp_inf.append(outpline)
            for line in outp_inf:
                outf.write(line)

def calc_angle():
    u = mda.Universe(STRUCTFNAME, TRAJFILE)
    ps = u.atoms.select_atoms("name P")
    c1s = u.atoms.select_atoms("name C1")
    c2s = u.atoms.select_atoms("name C2")
    c3s = u.atoms.select_atoms("name C3")
    with open(OUTPUTFILENAME, "w") as outf:
        outf.write("{: <15}{: <10}{: <30}{: <30}\n".format("time", "resid", "cosgamma", "cosdelta"))
        for ts in u.trajectory:
            LOGGER.info("at time %s", ts.time)
            outp_inf = []
            for P, C1, C2, C3 in zip(ps, c1s, c2s, c3s):
                PC1 = P.position - C1.position
                gamma = np.dot(PC1, np.array([0, 0, 1])) / np.linalg.norm(PC1)
                delta = np.dot(PC1, np.array([1, 0, 0])) / np.linalg.norm(PC1)
                outp_line = "{: <15}{: <10}{: <30}{: <30}\n".format(ts.time, P.residue.resid, gamma, delta)
                outp_inf.append(outp_line)
            for line in outp_inf:
                outf.write(line)


if __name__ == '__main__':
    if MODE == "distance":
        calc_distance()
    elif MODE == "angle":
        calc_angle()
    else:
        print("ERROR: Unknown mode")
        sys.exit()
