#!/usr/bin/env bash

set -e

echo "Total Perspective Vortex - EEG motor imagery classification"

if ! command -v python3 &> /dev/null; then
    echo "Python3 not found. Please install it."
    exit 1
fi

# pip package:python module
COMPONENTS=(
    scikit-learn:sklearn
    mne:mne
    scipy:scipy
    numpy:numpy
    matplotlib:matplotlib
    PyQt5:PyQt5
)

FILE="./nbr_pkg.txt"
required_count=${#COMPONENTS[@]}

if [ -f "$FILE" ]; then
    export NUMBER_OF_PKG=$(cat "$FILE")
else
    NUMBER_OF_PKG=0
fi

if [ "${NUMBER_OF_PKG:-0}" -ne "$required_count" ]; then
    number_of_pkg=0
    for component in "${COMPONENTS[@]}"; do
        package="${component%%:*}"
        module="${component##*:}"
        if ! python3 -c "import $module" &> /dev/null; then
            echo "Installing $package..."
            python3 -m pip install "$package" >/dev/null
        else
            echo "$package already installed"
        fi
        ((++number_of_pkg))
    done
    echo "Python packages verified/installed: $NUMBER_OF_PKG"
    echo "$number_of_pkg" > "$FILE"
else
    echo "All Python packages already verified ($NUMBER_OF_PKG total). Skipping check."
fi
