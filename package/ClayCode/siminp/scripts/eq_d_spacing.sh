gmx=GMX
numformat () {
	if [ ${#1} -le 1 ]; then # if first digit is less than or equal to 1 add a zero at the start when assigning variable numstr
		numstr=0$1
	else
		numstr=$1
	fi
}

i=3 # i = run number
n=8 # n = total number of runs
spacing=0 # waters
ingro=INGRO
intop=INTOP
remove_waters=REMOVEWATERS
mdp=MDP
numformat $i
namei=eq_d_spacing_$numstr

while [[ $i -lt $n ]]; do # while run number is less than total runs AND spacing equals zero
	mkdir -p ${namei}
	if [ $i -eq 0 ]; then # if run number is zero copy files
		echo Copying ${ingro} to ${namei}/${namei}.gro
		cp ${ingro} ${namei}/${namei}.gro
		echo Copying ${intop} to ${namei}/${namei}.top
		cp ${intop} ${namei}/${namei}.top
	fi
	echo Checking d-spacing for run $i
	spacing=$(python3 DSPACESCRIPT -p ${namei}/${namei} -n ${remove_waters} -d "TARGETSPACING") # script: run_number, run_name, num_water_to_remove, target_d_spacing,

	if [[ $spacing -eq 0 ]]; then # if remove_waters ran successfully:
		i=`expr $i + 1`
		numformat $i
		namei=eq_d_spacing_$numstr
		echo Preparing run $i # equilibrate the new system
		echo Running simulation $namei
		${gmx} grompp -f ${mdp} -p ${namei}/${namei}.top -c ${namei}/${namei}.gro -o ${namei}/${namei}.tpr -po ${namei}/${namei}.mdp
		${gmx} mdrun -v -s ${namei}/${namei}.tpr -deffnm ${namei}/${namei} MDRUNPRMS
	else
		printf "\nCompleted simulation run: ${namei}\nprevious run: ${namep}"
		echo "Reached target d-spacing of ${spacing}\n"
		echo "Writing ${namei}/${namei}.gro to ../eq_d_spacing.gro"
		cp ${namei}/${namei}.gro ../eq_d_spacing.gro
		break
	fi
	namep=$namei
done
