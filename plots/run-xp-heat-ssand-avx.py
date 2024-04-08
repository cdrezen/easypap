#!/usr/bin/env python3
from expTools import *

# Recommanded plot :
# ./plots/easyplot.py -if heat-ssand-avx.csv --plottype heatmap -heatx tilew -heaty tileh -v omp_tiled -- row=schedule aspect=1.8

easypapOptions = {
    "-k": ["ssandPile"],
    "-i": [128],
    "-v": ["omp_tiled"],
    "-wt": ["avx"],
    "-s": [1024],
    "-of": ["heat-ssand-avx.csv"],
}

# OMP Internal Control Variable
ompICV = {
    "OMP_SCHEDULE": ["static"],
    "OMP_NUM_THREADS": [32],
    "OMP_PLACES": ["cores"],
}


easypapOptions["--label"] = ["rect"]
easypapOptions["-th"] = [1]
easypapOptions["-tw"] = [512]

execute("./run ", ompICV, easypapOptions, verbose=False, easyPath=".")

del easypapOptions["-th"]
del easypapOptions["-tw"]
easypapOptions["-th"] = [16]
easypapOptions["-tw"] = [32]

execute("./run ", ompICV, easypapOptions, verbose=False, easyPath=".")

easypapOptions["--label"] = ["rect1"]

del easypapOptions["-th"]
del easypapOptions["-tw"]
easypapOptions["-th"] = [8]
easypapOptions["-tw"] = [32]

execute("./run ", ompICV, easypapOptions, verbose=False, easyPath=".")

del easypapOptions["-th"]
del easypapOptions["-tw"]
easypapOptions["-th"] = [32]
easypapOptions["-tw"] = [16]

execute("./run ", ompICV, easypapOptions, verbose=True, easyPath=".")

# Lancement de la version seq avec le nombre de thread impose a 1

ompICV = {"OMP_NUM_THREADS": [1]}

del easypapOptions["-th"]
del easypapOptions["-tw"]
easypapOptions["-v"] = ["seq"]

execute("./run ", ompICV, easypapOptions, verbose=False, easyPath=".")


print("Recommended plot:")
print(" ./plots/easyplot.py -if heat-ssand-avx.csv --plottype heatmap", 
      " -heatx tilew -heaty tileh -v omp_tiled -- row=schedule aspect=1.8")
