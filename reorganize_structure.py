'''
    Reorganize structure file with given order of molecules
    Optional input of topology file
'''
import sys
import logging
import argparse

from collections import OrderedDict

import structreader as sr

######################### Class and function definitions ################

class Molecule:
    def __init__(self, resname, resid, molname=None):
        self.resname = resname
        self.resid = resid
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
        atmname = match["atm1"]
        atmnr = match["atm2"]
        crd = (match["x"], match["y"], match["z"])
        self.atomnames.append(atmname)
        self.atomnumbers.append(atmnr)
        self.coordinates.append(crd)

    def write_entry(self, file):
        ''' Write molecule entry to file '''
        for atm, atmnr, crd in zip(self.atomnames, self.atomnumbers, self.coordinates):
            outpstr = sr.GRO_PATTERN.format(self.resid, self.resname, atm, atmnr, crd)
            file.write(outpstr)

def within_prot_range(ndx, prot_ranges):
    for i, protinfo in prot_ranges:
        ndxrange = protinfo[1]
        if ndxrange[0] <= ndx <= ndxrange[0]:
            return i
    return None


def reorder_molecules(molecules, mol_order):
    '''
        reorders a list of Molecule() instances to mol_order
        Works just like sorted but with the more complex Molecule() objects
    '''
    outp_list = []
    sorted_lists = {}

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



def read_switch_file(file):
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
    return SWITCH_INFO

def change_molecules(molecules, switch_info):
    '''
        Changes resname of molecule group with a certain concentration
        using the switch info dictionary

        - beware of leaflets
        - switch names randomly
    '''

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
PARSER.add_argument('-m', action="store", nargs="?", metavar='mol_order',     required=False, help="Must be a comma separated list of residue names")
PARSER.add_argument('-p', action="store", nargs="?", metavar='topology file', required=False, help="Must be a gromacs type .top file")
PARSER.add_argument('-s', action="store", nargs="?", metavar='switch file', required=False, help="This file must contain information on how to switch molecules")

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
with open(GROFILE, "r") as GROFILE:
    nlines = len(GROFILE)
    resid_old = -1

    for i, line in enumerate(GROFILE):
        if (i > 1) and i != nlines-1:
            pass
        elif i == 0:
            header = line
        elif i == 1:
            total_atmnumber = line
        else:
            box_dim = line

        match_dict = sr.REGEXP_GRO.match(line).groupdict()
        resname    = match_dict["resn"]
        resid      = match_dict["resid"]
        atmnr      = match_dict["atm2"]

        ### New molecule ###
        protinfo_ndx = within_prot_range(atmnr, PROTNDX_RANGES)
        ### check if line belongs to protein
        if protinfo_ndx is not None:
            ### check if protein molecules is already added
            if molecules[-1].molname != PROTNDX_RANGES[protinfo_ndx][0]:
                molecules[-1].append(Molecule(resname, resid, molname=PROTNDX_RANGES[protinfo_ndx][0]))
        ### if new lipid resid
        elif resid_old != resid:
            molecules.append(Molecule(resname, resid))

        ### Add molecule info to the last molecule added ###
        molecules[-1].add_info(line)


### Write out reordered structure file ###
with open(OUTPUTFILE, "w") as OUTPUTFILE:
    OUTPUTFILE.write(header)
    OUTPUTFILE.write(total_atmnumber)
    if MOL_ORDER is not None:
        molecules = reorder_molecules(molecules, mol_order=MOL_ORDER)
    if SWITCH_INFO is not None:
        molecules = change_molecules(molecules, SWITCH_INFO)
    for mol in molecules:
        mol.write_entry(OUTPUTFILE)


### If topology file is parsed, write reordered topology file
if TOPFILE is not None:
    with open(TOPFILE, "r") as TOPFILE, open(TOPFILE.replace(".top", "_reordered.top")) as OUTTOP:
        identifier = None
        mol_entry_written = False
        for line in TOPFILE:

            if ";" in line:
                line, comment = line.split(";")[0], ";" + ''.join(line.split(";")[1:])
            else:
                comment = ""

            if "[" in line and "]" in line:
                identifier = line.replace("[", "").replace("]", "").strip()

            if identifier != "molecules":
                print(line, comment, file=OUTTOP)

            if identifier == "molecules":
                if mol_entry_written:
                    continue
                else:
                    write_topology_molecules(molecules, OUTTOP)
                    mol_entry_written = True
