numformat () {
if [ ${#1} -le 1 ]; then
numstr=0$1
else
numstr=$1
fi
}
run_count=0
max_run_count=MAX_RUN
remove_waters=REMOVE_WATERS
ingro=INGRO
intop=INTOP
odir=ODIR
mdp=MDP
gmx=GMX
numformat ${run_count}
outpname=eq_run${numstr}
while [[ ${run_count} -lt ${max_run_count} && ${remove_waters} -eq 0 ]];
do
  run_count=$(( run_count + 1 ))
  numformat ${run_count}
  outpname=eq_run${numstr}
  mkdir "${odir}/${outpname}"
  outgro="${odir}/${outpname}/${outpname}.gro"
  outtop="${odir}/${outpname}/${outpname}.top"
  dspace_check=$(DSPACE_SCRIPT ${ingro} ${outgro} DSPACE REMOVE_WATERS ${intop} "${outtop}")
  if [[ ${dspace_check} -eq 1 ]]; then
    echo "d-space equilibration finished"
  elif [[ ${dspace_check} -eq 2 ]]; then
    echo "d-space equilibration failed"
  else
    echo "d-space equilibration not finished"
    $gmx grompp -f ${mdp} -p ${intop} -c ${ingro} -o "${odir}/${outpname}/${outpname}.tpr" -po "${odir}/${outpname}/${outpname}.mdp"
    $gmx mdrun -v -s "${odir}/${outpname}/${outpname}.tpr" -deffnm "${odir}/${outpname}/${outpname}" -pin on -ntomp 32 -ntmpi 1
  fi
  ingro="${outgro}"
  intop="${outtop}"
done
