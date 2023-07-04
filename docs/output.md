Output files (inside `<OUTPATH>` directory)

1. Sheet coordinates `<SYSNAME>_<X_CELLS>_<Y_CELLS>_<sheetnumber>.gro`
2. Interlayer coordinates and topology (with solvent and/or ions) (if interlayer is solvated and/or clay has non-zero charge) `<SYSNAME>_<X_CELLS>_<Y_CELLS>_interlayer.gro` and `<SYSNAME>_<X_CELLS>_<Y_CELLS>_interlayer.top`
3. (Solvated) clay sheet stack coordinates and topology `<SYSNAME>_<X_CELLS>_<Y_CELLS>.gro` and `<SYSNAME>_<X_CELLS>_<Y_CELLS>.top`
4. Sheet stack coordinates and topology in extended box (if specfied box height > height of clay sheet stack) `<SYSNAME>_<X_CELLS>_<Y_CELLS>_ext.gro` and `<SYSNAME>_<X_CELLS>_<Y_CELLS>_ext.top`
5. Sheet stack coordinates and topology in extended and solvated box (if bulk solvation is specified and specfied box height > height of clay sheet stack) `<SYSNAME>_<X_CELLS>_<Y_CELLS>_solv.gro` and `<SYSNAME>_<X_CELLS>_<Y_CELLS>_solv.top`
6. Sheet stack coordinates and topology in extended and solvated box with ions (if bulk solvation is specified and specified box height > height of clay sheet stack) `<SYSNAME>_<X_CELLS>_<Y_CELLS>_solv_ions.gro` and `<SYSNAME>_<X_CELLS>_<Y_CELLS>_solv_ions.top`
7. Sheet stack coordinates and topology in box after energy minimisation `<SYSNAME>_<X_CELLS>_<Y_CELLS>_<sheetnumber>_em.gro` and `<SYSNAME>_<X_CELLS>_<Y_CELLS>_<sheetnumber>_em.top`
8. Sheet stack energy minimisation parameter file and log`<SYSNAME>_<X_CELLS>_<Y_CELLS>_<sheetnumber>_em.mdp` and `<SYSNAME>_<X_CELLS>_<Y_CELLS>_<sheetnumber>_em.log`
9. Setup log file `<SYSNAME>_<X_CELLS>_<Y_CELLS>_<sheetnumber>.log`

