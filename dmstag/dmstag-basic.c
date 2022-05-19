#include <petsc.h>

static const char *const help = "Create a DMStag object.\n\n";

int main(int argc, char *argv[]) {
    PetscMPIInt rank;
    DM stag;
    PetscInt xs, ys, xm, ym, nex, ney;
    Vec v;
    PetscReal ***arrv;
    PetscInt ip, iex, iey, ie;
    PetscInt i, j;

    /* Initialize PETSc. */
    PetscInitialize(&argc, &argv, NULL, help);

    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

    /* Create DMStag. */
    DMStagCreate2d(
        PETSC_COMM_WORLD,   /* MPI communicator. */
        DM_BOUNDARY_NONE,
        DM_BOUNDARY_NONE,   /* Type of ghost nodes in the x- and y-direction.
                               Here, no ghost node exists. */
        2, 2,               /* Global number of elements in the x- and y-
                               direction. */
        PETSC_DECIDE,
        PETSC_DECIDE,       /* Number of processes in the x- and y- direction.
                               Let PETSc decide it. */
        1, 1, 1,            /* Degree of freedom (the number of variables
                               defined) per a point, an edge, and an element. */
        DMSTAG_STENCIL_STAR,/* Stencil type. */
        1,                  /* Stencil width. */
        NULL, NULL,         /* Arrays of the number of nodes in each process
                               along the x- and y-directions. Since PETSc
                               decides it, set these to NULL. */
        &stag
    );
    DMSetFromOptions(stag);
    DMSetUp(stag);

    /* Get the local part of the grid. */
    DMStagGetCorners(
        stag,
        &xs, &ys, NULL,     /* Index of the first local element. */
        &xm, &ym, NULL,     /* Number of local elements in each direction. */
        &nex, &ney, NULL    /* Number of extra elements in each direction. Since
                               the number of points are larger than the number
                               of elements by 1, there must be extra elements
                               beyond the last local elements to access to the
                               last local points, except for a periodic case. */
    );
    /* Print the local elements. */
    PetscSynchronizedPrintf(
        PETSC_COMM_WORLD,
        "Process %d has local elements [%d, %d] x [%d, %d]\n",
        rank, xs, xs + xm - 1, ys, ys + ym - 1
    );
    PetscSynchronizedFlush(PETSC_COMM_WORLD, PETSC_STDOUT);

    /* Create a local vector on the DMStag. */
    DMCreateLocalVector(stag, &v);
    /* Get the internal array of the vector. Since the grid is 2D, the array
       must be 3D. The first index corresponds to the index in y-direction, the
       second in x-direction, and the last corresponds to the variable index. To
       get the variable index, use DMStagGetLocationSlot. */
    DMStagVecGetArray(stag, v, &arrv);
    /* Get the index of the variable on a grid point. Locations of the DMStag is
       expressed as an orientation relative to an element. Note that a grid
       point (i,j) is at down left of an element (i,j). 0 in the third parameter
       means the first variable defined on a grid point. */
    DMStagGetLocationSlot(stag, DMSTAG_DOWN_LEFT, 0, &ip);
    /* Get the index of the variable on an edge perpendicular to x-direction. */
    DMStagGetLocationSlot(stag, DMSTAG_LEFT, 0, &iex);
    /* Get the index of the variable on an edge perpendicular to y-direction. */
    DMStagGetLocationSlot(stag, DMSTAG_DOWN, 0, &iey);
    /* Get the index of the variable on an element. */
    DMStagGetLocationSlot(stag, DMSTAG_ELEMENT, 0, &ie);

    /* Set the values of the vector. Setting values of extra elements or edges
       is available, but meaningless. */
    for (j = ys; j < ys + ym + ney; j++)
        for (i = xs; i < xs + xm + nex; i++) {
            arrv[j][i][ip] = 1;
            arrv[j][i][iex] = 2;
            arrv[j][i][iey] = 3;
            arrv[j][i][ie] = 4;
        }

    /* Restore the array. */
    DMStagVecRestoreArray(stag, v, &arrv);

    /* Destory. */
    DMDestroy(&stag);
    VecDestroy(&v);

    /* Finalize PETSc. */
    PetscFinalize();
}
