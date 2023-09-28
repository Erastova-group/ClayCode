#!/bin/bash

# Using AWK and bash, identify the clay sheets in a .gro file and,
# if they are present, remove a defined number of iSL water molecules from
# a .gro file if the average z-spacing between the adjacent clay sheets
# is above a defined threshold. Clay sheets are continuous groups of atoms
# identified from ClayFF atom types taken from YAML file keys or same residue
# names as previous matches.
# The script is designed to be used in a loop to remove water molecules until
# the average z-spacing between the clay sheets is below a defined threshold.

# Usage: remove_waters.sh <input.gro> <output.gro> <threshold> <number of waters to remove> <YAML file>

# Example: remove_waters.sh input.gro output.gro 1.0 1 clayff.yaml

INPUT=$1
OUTPUT=$2
THRESHOLD=$3
NUMWATERS=$4
YAML=$5

# get the average z-spacing between the clay sheets

# get the atom types from keys in YAML file and make all upper case
ATOMTYPES=$(grep '^[a-zA-Z]' "$YAML_FILE" | cut -c 1-3 | tr 'a-z' 'A-Z' | awk '{ printf "\"%s\",", $0 }' | sed 's/,$//')

# print the atom types to the screen
echo "Atom types: $ATOMTYPES"
