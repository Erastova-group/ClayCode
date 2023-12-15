#!/bin/bash

#check_env_exists() {
#  env_check=$(conda search --envs -p ${env_path} --json)
#}
#
#path=$(pwd -P)
#env_path=${path}/setup/ClayCode-env
#python_version=$(sed -n -E 's/(requires-python = )("[><~]?=)([0-9.]+)"/\3/p' pyproject.toml)
#
#check_env_exists
#if [[ $1 == '--clean' ]]
#then
#  echo "$1: Preparing for new environment setup"
#  conda env remove -p setup/claycode-conda-env
#fi
#
#check_env_exists
#if [[ ${env_check} == [] ]]
#then
#  echo "Creating new python ${python_version} environment in '${env_path}'"
#  conda create -p ${env_path} python=${python_version}
#fi
#
#conda shell.bash activate ${env_path} 1> /dev/null
#
#echo "Installing ClayCode in '${env_path}' environment"
#pip install . 1> /dev/null
#
#echo "Done!"
#
#ClayCode


