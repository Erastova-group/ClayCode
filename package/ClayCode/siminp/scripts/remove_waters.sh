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

INGRO="$1"
OUTGRO="$2"
THRESHOLD=$3
NUMWATERS=$4
YAML="$5"
INTOP="$6"
OUTTOP="$7"
GMX="$8"
ODIR="$9"
OUTPNAME="${10}"
MDP="${11}"


echo "Input gro file: $INGRO"
echo "Output gro file: $OUTGRO"
echo "Input top file: $INTOP"
echo "Output top file: $OUTTOP"
echo "Threshold: $THRESHOLD"
echo "Number of waters to remove: $NUMWATERS"
echo "YAML file: $YAML"


# get the average z-spacing between the clay sheets

# get the atom types from keys in YAML file and make all upper case
ATOMTYPES=$(grep -o "^[a-zA-Z2]\+" $YAML | cut -c 1-3 | tr '[:lower:]' '[:upper:]' | awk '{ printf "%s, ", $0 }' | sed 's/,$//')
# print the atom types to the screen
echo "Looking for clay atom types: $ATOMTYPES"


# Check if the GRO file has iSL residues
if grep -q "iSL" $INGRO; then
  remove_wat=$(awk -v threshold="$THRESHOLD" -v atomtypes="$ATOMTYPES" -v numwat="$NUMWATERS" -F "[-,]" '
  function abs(x) {return x<0 ? -x : x}
  BEGIN {
  FS=" +";
  split(atomtypes, atomtypes_array, ", ");
  sum_z = 0;
  num_iSL = 0;
  num_clay_sheets = 0;
  num_clay_molecules = 0;
  prev_res = "";
  prev_iSL = "";
  sheet_array[0] = 0;
  n_ucs = 0;
  final = 0;
  max_n_ucs = -1;
  max_n_atoms = -1;
  n_iSL = 0;
  max_n_iSL = -1;
  iSL_array[0] = 0;
  n_il = 0;
  }
  {
  if (NR == 1) { sysname = $0; next ; }
  if (NR == 2) { tot_atoms = $1; next; }
  # use awk to get the residue numbers of atoms matching the atom names in INFILE
  # in the atomtypes_array
  # print the residue numbers to the screen
  atomname = substr($0, 11, 5);
  resnum = substr($0, 1, 5);
  resname = substr($0, 6, 5);
  z_coord = $7;
  # get clay atoms and save their indices and z-coordinates
  clay_sheets = 0;
  match_found = 0;
  for (i in atomtypes_array) {
    if (atomtypes_array[i] != "" && index(atomname, atomtypes_array[i])) {
      match_found = 1;
      break;
      }}
    # find line of last clay match
    if (match_found == 1) {
      if (resnum != prev_res) {
        if (prev_res != "") {
          sheet_array[num_clay_sheets - 1 ] += sum_z/num_clay_atoms;
        }
        if ((resnum - prev_res != 1 && prev_res != "")) {
#          print "\tOld sheet ending with ", prev_res + 1;
#          print "\tNew clay sheet ", num_clay_sheets + 1, " starting from ", resnum;
          sheet_array[num_clay_sheets - 1 ] /= n_ucs;
          num_clay_sheets++;
          max_n_ucs = n_ucs;
          n_ucs=0;
        }
        n_ucs++;
        sum_z = z_coord;
        max_n_atoms = num_clay_atoms;
        num_clay_atoms = 1;
      } else {
        sum_z += z_coord;
        num_clay_atoms++;
      }
      prev_res = resnum;
    }
    if ( resname ~/iSL/ ) {
      if ( resnum - prev_iSL == 1 && prev_iSL != "") {
        num_iSL++;
#        if (max_n_iSL - num_iSL  == 1) {
#          iSL_array[n_il] = resnum;
#          print "Last Resnum: ", resnum, " ", num_iSL;
#          n_il++;
#          }
      } else
      if ( resnum - prev_iSL > 1 || prev_iSL == "" ) {
        num_iSL++;
        if (num_iSL > max_n_iSL) {
          if (prev_iSL != "") {
            max_n_iSL = num_iSL;
          }
          iSL_array[n_il] = resnum;
#          print "Resnum: ", resnum, " ", num_iSL;
          n_il++;
          }
          num_iSL = 1;
        }
        prev_iSL = resnum;
      }
  }
  END {
  sheet_array[num_clay_sheets - 1 ] += z_coord;
        sheet_array[num_clay_sheets - 1 ] += sum_z/num_clay_atoms;
        sheet_array[num_clay_sheets - 1 ] /= n_ucs;
        for (i = 0; i <= num_clay_sheets; i++) {
          distance[i] = "";
          for (j = 0; j <= num_clay_sheets; j++) {
            if (i != j) {
              new_distance = abs(sheet_array[i] - sheet_array[j]);
              if (distance[i] == "" || new_distance < distance[i]) {
              distance[i] = new_distance;
              };
              }
            }
          }
  max_distance = -1;
  max_distance_id = -1;
  for (i in distance) {
    if (max_distance_id == -1 || distance[i] > max_distance) {
      max_distance_id = i;
      max_distance = distance[i];
      }
    }
  print "Max distance: ", max_distance;
  delete distance[max_distance_id];
  mean_distance = 0;
  for (i in distance) {
  mean_distance += distance[i];}
  mean_distance /= num_clay_sheets;
  print("Distance between clay sheets: ", mean_distance);
  print "Number of clay sheets: ", num_clay_sheets + 1;
  print "Number of clay UCs per sheet: ", max_n_ucs;
  print "Number of clay atoms per UC: ", max_n_atoms;
  print "Number of iSL residues: ", num_iSL;
  print "Average z-spacing between clay sheets: ", mean_distance, " nm";
  print "Maximum target distance between clay sheets: ", threshold, " nm";
  if (mean_distance > threshold)
  {
    if (num_iSL < numwat)
    {
      print "Not enough iSL (", numwat,") waters to remove from total of ", num_iSL;
      numwat = num_iSL;
    }
    else
    {
      print "BEGIN REMOVE_WATERS";
      for (i in iSL_array)
      {
        if (iSL_array[i] != "")
        {
          print iSL_array[i], " ", numwat;
        }
      }
      print "END REMOVE_WATERS";
    }
  }
  else
  {
    print "No waters to remove";
  }
}' "$INGRO")
  remove_waters_true=$(sed -n '/BEGIN REMOVE_WATERS/,/END REMOVE_WATERS/p' <<< "$remove_wat" )
  printf "$remove_waters_true"
  if [[ "$remove_waters_true" != "" ]]; then
    cp -u $INGRO $OUTGRO
    log_str=$(sed -n '/BEGIN REMOVE_WATERS/q;p' <<< "$remove_wat")
    echo "$log_str"
    il_count=$(sed -n '/Number of clay_sheets:\s\+\([0-9]\+\)/s//\1/p' <<< "$log_str")
    isl_count=$(sed -n '/Number of iSL residues:\s\+\([0-9]\+\)/s//\1/p' <<< "$log_str")
    # get column 1 and 2 from awk output after "Removing waters from interlayers:" to get residues and number of waters to remove
#    remove_pattern=$(sed '/Removing waters from interlayers/,$!d' <<< "$remove_wat" | awk '{print $1}' | sed -E -n 's/([0-9]+)$/\1/p')
    remove_pattern=$(awk '{print $1}' <<< "$remove_waters_true" | sed -E -n 's/([0-9]+)$/\1/p')
#    echo "Removing $remove_pattern waters from interlayers" #-F "[-,]" '
    remove_str=$(awk -v remove_pattern="$remove_pattern" -v numwat="$NUMWATERS" '
#    function substitute(pattern, replace, string)
#    {
#      match(string, pattern);
#      pre_match = substr(string, 1, RSTART - 1);
#      post_match = substr(string, RSTART + RLENGTH);
#      print post_match;
#      return pre_match replace post_match;
#    }
    BEGIN {
    OFS=FS=" ";
#    ORS="\n";
    split(remove_pattern, remove_array, "\n");
    lines[0] = 0;
    n_remove_lines = 0;
    n_remove_atoms = 0;
    n_remove_molecules = 0;
    prev_resname = "";
    }
    {
      if (NR == 1) { sysname = $0; outfile[NR]=$0 ; next;}
      if (NR == 2) { tot_atoms = $1; outfile[NR]=$0 ; next;}
      if ($0 ~ "^[0-9 \t.]+$") {dimensions = $0; outfile[NR]=$0 ; next ;}
      resnum = substr($0, 1, 5);
      resname = substr($0, 6, 5);
      atomnum = substr($0, 16, 5);
      atomname = substr($0, 11, 5);
#      print atomname;
      if (resname ~/iSL/)
      {
        for (i in remove_array)
        {
          for (j = remove_array[i]; j < remove_array[i] + numwat; j++)
          {
            if (resnum ~ "[ \t]+" j "[ \t]*$")
            {
              lines[n_remove_lines] = NR;
              n_remove_lines++;
              n_remove_atoms++;
              if (prev_resnum != resnum)
              {
                n_remove_molecules++;
                prev_resnum = resnum;
              }
            }
          }
        }
      }
    if (n_remove_molecules == 0)
    {
        outfile[NR] = $0;
    }
    else
    {
      new_resnum = resnum - n_remove_molecules;
      new_atomnum = atomnum - n_remove_atoms;
      new_str = $0;
        sub(/ */, "", resnum);
        sub(resnum, new_resnum, new_str);
        sub(/ */, "", atomnum);
        sub(atomname " *" atomnum, atomname new_atomnum, new_str);
#      a=sub(/( 0)([0-9]+)[a-zA-Z]+[0-9 ]+[a-zA-Z]+[0-9]? *[0-9]+ .*(0)/, "\1" new_resnum "\2" new_atomnum " \3", new_str);
      outfile[NR] = new_str;
    }
    }
    END {
    tot_atoms-=n_remove_lines;
    for (i = 0; i < length(lines); i++){
      outfile[lines[i]] = "";
#      print "Removing line", lines[i], "/", FNR, "\n";
      }
    print "Removed ", n_remove_molecules, "iSL molecules (",n_remove_lines, "atoms from", tot_atoms, "atoms in system )";
#    print "Removed", n_remove_atoms, "atoms from", tot_atoms, "atoms in system"
    outfile[2] = tot_atoms - n_remove_atoms;
    print "BEGIN GRO";
    for (i = 0; i < length(outfile); i++)
    {
      if (outfile[i] != "")
      {
        print outfile[i];
      }
    }
#    print dimensions;
    print "END GRO";
    } ' $OUTGRO)

    new_file=$(sed -n '/BEGIN GRO/,/END GRO/p' <<< "$remove_str" | sed '/BEGIN GRO/d' | sed '/END GRO/d')
    log_str=$(sed -n '/BEGIN/q;p' <<< "$remove_str")
    printf "$log_str"
    echo "$new_file" > "${OUTGRO}"
    echo Wrote new coordinates to $OUTGRO

    # Modify the topology file to remove the iSL residues after [ system ]

    header=$(sed -n '1,/\[ molecules \]/p' $INTOP)
    new_nwat=$((isl_count - NUMWATERS))
    system_str=$(sed -n '/\[ molecules \]/,$ {/\[ molecules \]/!p}' $INTOP | sed "s/\(iSL\s\+\)\([0-9]\+\)\(\s*$\)/\1${new_nwat}\3/")
    printf "$header\n$system_str" > $OUTTOP
    echo "New interlayers have ${new_nwat} iSL residues in interlayers"
    echo Wrote new topology to $OUTTOP
    if [[ "${12}" == '--debug' ]]; then
      echo $GMX grompp -f "${MDP}" -p ${OUTTOP} -c ${OUTGRO} -o "${ODIR}/${OUTPNAME}/${OUTPNAME}.tpr" -po "${ODIR}/${OUTPNAME}/${OUTPNAME}.mdp" -pp "${ODIR}/${OUTPNAME}/${OUTPNAME}.top"
      echo $GMX mdrun -v -s "${ODIR}/${OUTPNAME}/${OUTPNAME}.tpr" -deffnm "${ODIR}/${OUTPNAME}/${OUTPNAME}" -pin on -ntomp 32 -ntmpi 1
      cp $OUTGRO "${ODIR}/${OUTPNAME}/${OUTPNAME}.gro"
      cp $OUTTOP "${ODIR}/${OUTPNAME}/${OUTPNAME}.top"
    else
      $GMX grompp -f "${MDP}" -p ${OUTTOP} -c ${OUTGRO} -o "${ODIR}/${OUTPNAME}/${OUTPNAME}.tpr" -po "${ODIR}/${OUTPNAME}/${OUTPNAME}.mdp" -pp "${ODIR}/${OUTPNAME}/${OUTPNAME}.top"
      $GMX mdrun -v -s "${ODIR}/${OUTPNAME}/${OUTPNAME}.tpr" -deffnm "${ODIR}/${OUTPNAME}/${OUTPNAME}" -pin on -ntomp 32 -ntmpi 1

    fi
    exit 0
  else
    echo "Not removing iSL residues from $INGRO"
    exit 1
  fi
  else
    echo "No iSL residues found in $INGRO"
    exit 2
fi
