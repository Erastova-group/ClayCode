#!/bin/python3

import itertools
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# p = Path(sys.argv[1])
p = Path("/media/hannahpollak/free")
linestyle = itertools.cycle(["-", "--", "-.", ":"])


fig, ax = plt.subplots(1, 2, figsize=(40, 10))

files = p.glob("*z_group_[0-9].dat")

for file in sorted(files):
    df = pd.read_csv(
        file,
        comment="#",
        sep="\s+",
        header=None,
        names=["r", "g(r) abs", "g(r) rel"],
    )
    ax[0].plot(
        df["r"],
        df["g(r) rel"],
        label=file.stem,
        linestyle=linestyle.__next__(),
    )

df = pd.read_csv(
    p / "claycode_analysis_tests_rdf.dat",
    comment="#",
    sep="\s+",
    header=None,
    names=["r", "g(r) abs", "g(r) rel"],
)
ax[0].plot(
    df["r"],
    df["g(r) rel"],
    label="claycode_analysis_tests_rdf",
    linestyle=linestyle.__next__(),
)

rfiles = p.glob("*z_group_[0-9]_crd_numbers.dat")
for fnum, file in enumerate(sorted(rfiles)):
    df = pd.read_csv(
        file,
        comment="#",
        sep="\s+",
        header=None,
        names=["r", "g(r) abs", "g(r) rel"],
    )
    ax[1].bar(
        df["r"] + fnum * (df["r"].sum()),
        df["g(r) rel"],
        label=file.stem,
        linestyle=linestyle.__next__(),
    )

# df = pd.read_csv(p / "claycode_analysis_tests_crd_numbers.dat", comment="#", sep="\s+", header=None, names=["r", "g(r) abs", "g(r) rel"])
# ax[1].bar(df["r"], df["g(r) rel"], label="claycode_analysis_tests_rdf", linestyle=linestyle.__next__())

ax[0].legend()
ax[1].legend()
plt.show()
