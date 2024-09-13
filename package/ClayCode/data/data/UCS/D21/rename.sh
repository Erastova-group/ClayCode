#!/bin/bash

grofiles=$(ls D2{0..9}{0..9}.gro)

for file in ${grofiles[@]}; do
        new_file=${file%.gro}
        name=${new_file}
        stem=${new_file#D2}
        new_file=D20${stem}.gro
        echo ${new_file} $file
        cp $file $new_file
        sed -E -i 's/(\s1+)D2([0-9]{3})/\1D20\2/g' ${new_file}
        new_file=D20${stem}.itp
        cp ${name}.itp ${new_file}
        sed -i "s/${name}/D20${stem}/g" ${new_file}
done
