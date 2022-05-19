/*
DM (Data Management) manages a topolgical structure such as a mesh, tree, or
graph. There are several types of DM:
  - DMDA (Distributed Array) is for a collocated cartesian grid
  - DMStag is for a staggered cartesian grid
  - DMPlex is for a unstructured grid
  - etc.

This code creates a 2D structured grid with DMDA and prints each process' local
part of the grid.
*/

#include <petsc.h>

static const char *const help = "Create a DMDA object.\n\n";

int main(int argc, char *argv[]) {
    PetscMPIInt rank;
    DM da;
    DMDALocalInfo info;
    PetscInt stencil_width = 1;

    /* Initialize PETSc. */
    PetscInitialize(&argc, &argv, NULL, help);

    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

    PetscOptionsGetInt(NULL, NULL, "-stencil_width", &stencil_width, NULL);

    /* Create the DMDA object of global size 3 x 3. */
    DMDACreate2d(
        PETSC_COMM_WORLD,   /* MPI communicator. */
        DM_BOUNDARY_NONE,
        DM_BOUNDARY_NONE,   /* Type of ghost nodes in the x- and y-direction.
                               Here, no ghost node exists. */
        DMDA_STENCIL_STAR,  /* Stencil type. If each node depends on the nodes
                               with same x, y, or z coordinate only, use
                               DMDA_STENCIL_STAR. Otherwise, use
                               DMDA_STENCIL_BOX. */
        3, 3,               /* Global number of grid points in the x- and y-
                               direction. Command line options -da_grid_{x,y,z}
                               and -da_refine can change this value. */
        PETSC_DECIDE,
        PETSC_DECIDE,       /* Number of processes in the x- and y- direction.
                               Let PETSc decide it. */
        1,                  /* Degree of freedom. Mostly 1. */
        stencil_width,      /* Stencil width. The number of ghost nodes each
                               process has depends on this value. */
        NULL, NULL,         /* Arrays of the number of nodes in each process
                               along the x- and y-directions. Since PETSc
                               decides it, set these to NULL. */
        &da
    );
    DMSetFromOptions(da);
    DMSetUp(da);

    /* Get the local part of the grid. */
    DMDAGetLocalInfo(da, &info);
    /* Print the part. */
    PetscSynchronizedPrintf(PETSC_COMM_WORLD,
                            "Process %d has local part [%d, %d] x [%d, %d]\n",
                            rank, info.xs, info.xs + info.xm - 1, info.ys,
                            info.ys + info.ym - 1);
    PetscSynchronizedFlush(PETSC_COMM_WORLD, PETSC_STDOUT);

    /* Destroy DMDA. */
    DMDestroy(&da);

    /* Finalize PETSc. */
    PetscFinalize();
}
