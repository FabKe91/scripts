#!/bin/env python3
'''
    Reorganize structure file with given order of molecules
    Optional input of topology file
'''
import sys
import logging
import argparse

import MDAnalysis as mda

from random import shuffle
from collections import OrderedDict

import structreader as sr

######################### Class and function definitions ################

class Molecule:
    def __init__(self, resname, molname=None):
        self.resnames = []
        self.resids = []
        self.coordinates = []
        self.atomnames = []
        self.atomnumbers = []
        self.lines = []

        if molname is not None:
            self.molname = molname
        else:
            self.molname = resname

    def add_info(self, line):
        ''' Add structure info from line in structure file '''
        self.lines.append(line)
        match = sr.REGEXP_GRO.match(line).groupdict()
        atmname = match["atm1"].strip()
        atmnr   = match["atm2"].strip()
        resname = match["resn"].strip()
        resid   = match["resid"].strip()
        crd = (match["X"], match["Y"], match["Z"])
        self.atomnames.append(atmname)
        self.atomnumbers.append(atmnr)
        self.coordinates.append(crd)
        self.resnames.append(resname)
        self.resids.append(resid)

    def write_entry(self, file):
        ''' Write molecule entry to file '''
        for resname, resid, atm, atmnr, crd in\
            zip(self.resnames, self.resids, self.atomnames, self.atomnumbers, self.coordinates):
            outpstr = sr.GRO_STRING_FORMAT.format(resid, resname, atm, atmnr, *[float(i) for i in crd])
            file.write(outpstr+'\n')

    def update_resname(self, resname):
        self.resnames = [resname for _ in self.resnames]
    def update_molname(self, molname):
        self.molname = molname

def assign_leaflets(grofilename):
    ''' '''
    LOGGER.warning("WARNING: assigning leaflets according to refatoms PO4 so only for PLs!!!")
    leaflet_assignment = {}
    u = mda.Universe(grofilename)
    refatms = u.atoms.select_atoms("name PO4")
    center = refatms.center_of_geometry()[2]

    for atm in refatms:
        if atm.position[2] >= center:
            leaflet_assignment[atm.resid] = 1
        else:
            leaflet_assignment[atm.resid] = 0
    return leaflet_assignment


def within_prot_range(ndx, prot_ranges):
    ndx = int(ndx)
    for i, protinfo in prot_ranges:
        ndxrange = [int(i) for i in protinfo]
        if ndxrange[0] <= ndx <= ndxrange[1]:
            return i
    return None


def reorder_molecules(molecules, mol_order):
    '''
        reorders a list of Molecule() instances to mol_order
        Works just like sorted but with the more complex Molecule() objects
    '''
    outp_list = []
    sorted_lists = {}

    mol_order = mol_order.split(',')

    ### Sort lists to resname ###
    for mol in molecules:
        if mol.molname not in sorted_lists.keys():
            sorted_lists[mol.molname] = []
        sorted_lists[mol.molname].append(mol)

    ### rewrite list with order in mol_order ###
    for mol_ordername in mol_order:
        for mol in sorted_lists[mol_ordername]:
            outp_list.append(mol)
        del sorted_lists[mol_ordername]

    ### check wether all resnames were given ###
    if len(sorted_lists):
        raise ValueError("ERROR: Not all molecule types in system were parsed in molecule order"
                         "\tMissing molecule names:"
                         ' '.join(sorted_lists.keys())
            )
    return outp_list



def read_switch_file(filename):
    '''
        Will _randomly_ replace names of molecules in structure file

        Syntax of switch_info file:
            <src_resname> <target_resname> <conc_leaf1> <conc_leaf2>
        e.g.:
            DOPC DYPC 0.5 0.5
            DOPC YOPC 0.5 0.5
        changes all DOPC molecules to DYPC (one half) and YOPC (second half)

        switch_info is a dictionary
            SWITCH_INFO[<src_resname>] = [(<target_resname1>, <conc_leaf1>, <conc_leaf2>), ...]
        e.g.:
            SWITCH_INFO["DOPC"] = [("DYPC", 0.5, 0.5), ("YOPC", 0.5, 0.5)]

        concentration per leaflet must not be >1 !!!
    '''
    SWITCH_INFO = {}

    ### Read info and add to SWITCH_INFO dict ###
    with open(filename, "r") as file:
        for line in file:
            print(line)
            if line == "\n":
                continue
            srcname, targname, cl1, cl2 = line.split()
            if srcname not in SWITCH_INFO.keys():
                SWITCH_INFO[srcname] = []
            SWITCH_INFO[srcname].append((targname, cl1, cl2))

    ### Check wether input is valid ###
    for srcn, switchlist in SWITCH_INFO.items():
        cl1_sum = 0
        cl2_sum = 0
        for info in switchlist:
            cl1_sum += float(info[1])
            cl2_sum += float(info[2])
        if cl1_sum > 1 or cl2_sum > 1:
            print("ERROR: concentration value exceeds one with input")
            print("{}:\nsum conc1/conc2: {}/{}".format(srcname, cl1_sum, cl2_sum))
            sys.exit()

    return SWITCH_INFO

def change_lipids(molecules, switch_info, leaflet_assignment, mol_order=None):
    '''
        Changes resname of molecule group with a certain concentration
        using the switch info dictionary

        - beware of leaflets
        - switch names randomly
    '''
    LOGGER.warning("function change lipids only works for lipids.")
    srcnames_to_switch = switch_info.keys()
    count_dict= {moln:{0:0, 1:0} for moln in srcnames_to_switch}
    ndx_dict = {}
    ### get list of molecules to change
    ### separate list to leaflets
    for i, mol in enumerate(molecules):
        if mol.molname in srcnames_to_switch:
            if mol.molname not in ndx_dict.keys():
                ndx_dict[mol.molname] = {0:[], 1:[]}
            leafl = leaflet_assignment[int(mol.resids[0])]
            ndx_dict[mol.molname][leafl].append(i)
            count_dict[mol.molname][leafl] += 1
    ### change molecule info with correct ratio and randomly
    for srcmol, chng_info in SWITCH_INFO.items():
        for leafl in [0, 1]:
            shuffled_ndx = ndx_dict[srcmol][leafl].copy()
            shuffle(shuffled_ndx)
            len_ndx = len(shuffled_ndx)
            for targ, cl1, cl2 in chng_info:
                conc = [cl1, cl2][leafl]
                n_index_to_change = int(len_ndx*float(conc))
                ndx_to_change = shuffled_ndx[0:n_index_to_change]
                shuffled_ndx = shuffled_ndx[n_index_to_change:]
                for mol_ndx in ndx_to_change:
                    molecules[mol_ndx].update_resname(targ)
                    molecules[mol_ndx].update_molname(targ)

    return molecules

def write_topology_molecules(molecules, outputfile_obj):
    ''' Counts resnames in write topology entries in correct order '''
    mol_count = OrderedDict()
    for mol in molecules:
        if mol.molname not in mol_count.keys():
            mol_count[mol.molname] = 0
        mol_count[mol.molname] += 1

    for resn, amount in mol_count.items():
        outputfile_obj.write("{: <10}{: >15}\n".format(resn, amount))
    outputfile_obj.write("\n")


#########################################################################


#####################         LOGGER            #########################

LOGGER = logging.getLogger("plot_hofs")
LOGGER.setLevel("DEBUG")

# create console handler and set level to debug
ch = logging.StreamHandler()
ch.setLevel("INFO")
# create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# add formatter to ch
ch.setFormatter(formatter)
# add ch to logger
LOGGER.addHandler(ch)

#########################################################################


######################### ARGPARSE ARGUMENTS ############################

PARSER = argparse.ArgumentParser()

# Non optional parameters
PARSER.add_argument('-f', action="store", metavar='input_structure', required=True, help="Must be a .gro file")
PARSER.add_argument('-o', action="store", metavar='output file', required=True, help="output structure file name")
PARSER.add_argument('--protndx', action="store", nargs="*", metavar='protein_indexrange', required=True, help="range of tmd indices can be more than 1. syntax must be <protname>-<startindex>:<endindex>")
# optional arguments
#PARSER.add_argument('-l', '--lipid', action="store", nargs='?', metavar='PL type', required=False, default="DPPC",        help="Name of PL type (all caps)")
PARSER.add_argument('-m', action="store",  metavar='mol_order',     required=True, help="Must be a comma separated list of residue names")
PARSER.add_argument('-p', action="store",  metavar='topology file', required=True, help="Must be a gromacs type .top file")
PARSER.add_argument('-s', action="store",  metavar='switch file',   required=True, help="Syntax: <srcname> <targetname> <conc_l1> <conc_l2>, e.g. DOPC YOPC 0.5 0.5")

ARGS = PARSER.parse_args()

GROFILE    = ARGS.f
OUTPUTFILE = ARGS.o
### will be [(<protname>, [<startndx>, <endndx>]), (...), ...]
PROTNDX_RANGES = [(i.split("-")[0], i.split("-")[1].split(":")) for i in ARGS.protndx]

MOL_ORDER  = ARGS.m
TOPFILE    = ARGS.p
SWITCHFILE = ARGS.s

if GROFILE == OUTPUTFILE:
    print("ERROR: Input and output structure must be named differently")
    sys.exit()
if GROFILE[-3:] != "gro":
    print("ERROR: Input structure file not a .gro file?")
    sys.exit()
if TOPFILE is not None:
    if TOPFILE[-3:] != "top":
        print("ERROR: Input topology not a gromacs .top file?")
        sys.exit()

if SWITCHFILE:
    SWITCH_INFO = read_switch_file(SWITCHFILE)
else:
    SWITCH_INFO = None

#########################################################################

### Read grofile to gather all information about system ###
header          = None
total_atmnumber = None
box_dim         = None
molecules       = []

with open(GROFILE, "r") as gfile:
    ### Read entire file to get file length...
    for ndx, line in enumerate(gfile):
        nlines = ndx
    nlines += 1

    gfile.seek(0)

    resid_old = -1

    for i, line in enumerate(gfile):
        if (i > 1) and i != nlines-1:
            pass
        elif i == 0:
            header = line
            continue
        elif i == 1:
            total_atmnumber = line
            continue
        elif i == nlines - 1 :
            box_dim = line
            continue

        match_dict = sr.REGEXP_GRO.match(line).groupdict()
        resname    = match_dict["resn"].strip()
        resid      = match_dict["resid"].strip()
        atmnr      = match_dict["atm2"].strip()

        ### New molecule ###
        protname = within_prot_range(int(atmnr), PROTNDX_RANGES)
        ### check if line belongs to protein
        if protname is not None:
            ### check if protein molecules is already added
            if not molecules or molecules[-1].molname != protname:
                molecules.append(Molecule(resname, molname=protname))
        ### if new lipid resid
        elif resid_old != resid:
            molecules.append(Molecule(resname))

        ### Add molecule info to the last molecule added ###
        molecules[-1].add_info(line)
        resid_old = resid


### Write out reordered structure file ###
with open(OUTPUTFILE, "w") as OUTPUTFILE:
    OUTPUTFILE.write(header)
    OUTPUTFILE.write(total_atmnumber)
    if SWITCH_INFO is not None:
        leaflet_assignment = assign_leaflets(GROFILE)
        molecules = change_lipids(molecules, SWITCH_INFO, leaflet_assignment)
    if MOL_ORDER is not None:
        molecules = reorder_molecules(molecules, mol_order=MOL_ORDER)
    for mol in molecules:
        mol.write_entry(OUTPUTFILE)
    OUTPUTFILE.write(box_dim)


### If topology file is parsed, write reordered topology file
if TOPFILE is not None:
    with open(TOPFILE, "r") as tfile, open(TOPFILE.replace(".top", "_reordered.top"), "w") as outtop:
        identifier = None
        mol_entry_written = False
        for line in tfile:

            if ";" in line:
                line, comment = line.split(";")[0], ";" + ''.join(line.split(";")[1:])
            else:
                comment = ""

            if "[" in line and "]" in line:
                identifier = line.replace("[", "").replace("]", "").strip()
                print(line, file=outtop, end="")
            elif identifier != "molecules":
                print(line+comment, file=outtop, end="")

            if identifier == "molecules":
                if mol_entry_written:
                    continue
                else:
                    write_topology_molecules(molecules, outtop)
                    mol_entry_written = True
