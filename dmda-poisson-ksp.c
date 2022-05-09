/*
Consider a Poisson equation
    -(u_xx + u_yy) = 2x^3 + 6x(y^2-1)
on a unit sqaure [0, 1] x [0, 1] with the boundary condition
    u(0, y) = 0
    u(1, y) = -y^2 + 1
    u(x, 0) = x^3
    u(x, 1) = 0.
The exact solution is
    u(x, y) = -x^3(y^2-1).
This code solves the equation above using a finite difference method with a DMDA
and KSP.
*/

#include <petsc.h>

const char *const help
    = "Solve a 2D Poisson equation on a unit square with KSP.\n\n";

PetscErrorCode FormMatrix(DM, Mat);
PetscErrorCode FormRHS(DM, Vec);
PetscErrorCode FormExactSolution(DM, Vec);

int main(int argc, char *argv[]) {
    DM da;
    Mat A;
    Vec b, u, ut;
    KSP ksp;
    PetscReal e;

    /* Initialize PETSc. */
    PetscInitialize(&argc, &argv, NULL, help);

    /* Create a DMDA of default size 9 x 9. */
    DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,
                 DMDA_STENCIL_STAR, 9, 9, PETSC_DECIDE, PETSC_DECIDE, 1, 1,
                 NULL, NULL, &da);
    DMSetFromOptions(da);
    DMSetUp(da);

    /* Create a matrix on the DMDA. Setting the size and setting up are done in
       DMCreateMatrix() so MatSetUp() need not be called manually. */
    DMCreateMatrix(da, &A);
    /* Build the matrix. */
    FormMatrix(da, A);

    /* Create an RHS vector. */
    DMCreateGlobalVector(da, &b);
    /* Build the RHS vector. */
    FormRHS(da, b);

    /* Create and solve the linear system. */
    KSPCreate(PETSC_COMM_WORLD, &ksp);
    KSPSetOperators(ksp, A, A);
    KSPSetFromOptions(ksp);
    VecDuplicate(b, &u);
    KSPSolve(ksp, b, u);

    /* Calculate the maximum error. */
    VecDuplicate(u, &ut);
    FormExactSolution(da, ut);
    VecAXPY(ut, -1.0, u);
    VecNorm(ut, NORM_INFINITY, &e);
    PetscPrintf(PETSC_COMM_WORLD, "Maximum error: %g\n", e);

    /* Finalize PETSc. */
    PetscFinalize();
}

PetscErrorCode FormMatrix(DM da, Mat A) {
    DMDALocalInfo info;
    PetscReal hx, hy;
    MatStencil row, col[5];
    PetscReal v[5];
    PetscInt ncols;
    PetscInt i, j;

    /* Get the local information. */
    DMDAGetLocalInfo(da, &info);
    /* Get the grid spacing. */
    hx = 1.0 / (info.mx - 1);
    hy = 1.0 / (info.my - 1);

    /* Fill the matrix. */
    for (j = info.ys; j < info.ys + info.ym; j++)
        for (i = info.xs; i < info.xs + info.xm; i++) {
            /* Each point on the grid corresponds to a row of the matrix. Each
               row has 5 non-zero elements (except for a boundary point) since
               a 5-point stencil is used here. */
            row.i = i;
            row.j = j;
            col[0].i = i;
            col[0].j = j;

            if (i == 0 || i == info.mx - 1 || j == 0 || j == info.my - 1) {
                /* The value is given by the boundary condition; only one
                   non-zero element is. */
                v[0] = 1.0;
                ncols = 1;
            } else {
                v[0] = 2.0 / (hx * hx) + 2.0 / (hy * hy);
                col[1].i = i + 1;
                col[1].j = j;
                v[1] = -1.0 / (hx * hx);
                col[2].i = i - 1;
                col[2].j = j;
                v[2] = -1.0 / (hx * hx);
                col[3].i = i;
                col[3].j = j + 1;
                v[3] = -1.0 / (hy * hy);
                col[4].i = i;
                col[4].j = j - 1;
                v[4] = -1.0 / (hy * hy);
                ncols = 5;
            }

            /* Set value. */
            MatSetValuesStencil(A, 1, &row, ncols, col, v, INSERT_VALUES);
        }

    /* Assemble the matrix. */
    MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);

    return 0;
}

PetscErrorCode FormRHS(DM da, Vec b) {
    DMDALocalInfo info;
    PetscReal hx, hy;
    PetscReal **arrb, x, y;
    PetscInt i, j;

    /* Get the local information. */
    DMDAGetLocalInfo(da, &info);
    /* Get the grid spacing. */
    hx = 1.0 / (info.mx - 1);
    hy = 1.0 / (info.my - 1);

    /* Fill the vector. */
    DMDAVecGetArray(da, b, &arrb);
    for (j = info.ys; j < info.ys + info.ym; j++)
        for (i = info.xs; i < info.xs + info.xm; i++) {
            x = i * hx;
            y = j * hy;
            if (i == 0)
                arrb[j][i] = 0.0;
            else if (i == info.mx - 1)
                arrb[j][i] = -y*y + 1;
            else if (j == 0)
                arrb[j][i] = x*x*x;
            else if (j == info.my - 1)
                arrb[j][i] = 0.0;
            else
                arrb[j][i] = 2.0*x*x*x + 6.0*x*(y*y-1);
        }
    DMDAVecRestoreArray(da, b, &arrb);

    return 0;
}

PetscErrorCode FormExactSolution(DM da, Vec ut) {
    DMDALocalInfo info;
    PetscReal hx, hy;
    PetscReal **arrut, x, y;
    PetscInt i, j;

    /* Get the local information. */
    DMDAGetLocalInfo(da, &info);
    /* Get the grid spacing. */
    hx = 1.0 / (info.mx - 1);
    hy = 1.0 / (info.my - 1);

    /* Fill the exact solution. */
    DMDAVecGetArray(da, ut, &arrut);
    for (j = info.ys; j < info.ys + info.ym; j++)
        for (i = info.xs; i < info.xs + info.xm; i++) {
            x = i * hx;
            y = j * hy;
            arrut[j][i] = -x*x*x * (y*y - 1);
        }
    DMDAVecRestoreArray(da, ut, &arrut);

    return 0;
}
