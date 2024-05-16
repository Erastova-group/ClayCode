#!/bin/bash

# optional pipx arguments can be passed as the first argument
# e.g. ./install.sh --verbose --editable

# Remove existing installs
pipx uninstall ClayCodeTests > /dev/null 2>&1
pipx uninstall ClayCode > /dev/null 2>&1

# Install the package
pipx install --python python3.10 --force $1 .
