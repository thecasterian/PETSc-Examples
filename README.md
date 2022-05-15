PETSc-Examples
==============

Examples for PETSc.

Basics
------

Basics of PETSc: the structure of a program and three types of fundamental solvers.

* [petsc-basic.c](basic/petsc-basic.c) - Basic structure of a PETSc program.
* [ksp-basic.c](basic/ksp-basic.c) - Basic of the linear system solver (KSP).
* [snes-basic.c](basic/snes-basic.c) - Basic of the nonlinear system solver (SNES).

SNES
----

Advanced use and applications of the SNES.

* [snes-jacobian.c](snes/snes-jacobian.c) - SNES with user-specified Jacobian.
* [snes-linear.c](snes/snes-linear.c) - Solving a linear system with the SNES.

DMDA
----

Examples of the structured grid interface (DMDA).

* [dmda-basic.c](dmda/dmda-basic.c) - Basic of the DMDA.
* [dmda-vector.c](dmda/dmda-vector.c) - Vector operations on the DMDA.
* [dmda-poisson-ksp.c](dmda/dmda-poisson-ksp.c) - Solving Poisson equation with the KSP on the DMDA.
* [dmda-poisson-snes.c](dmda/dmda-poisson-snes.c) - Solving Poisson equation with the SNES on the DMDA.
