/*
Sometimes it is useful to solve a linear equation using SNES considering the
since SNES is easy to expand the code to a non-linear solver and provides more
options.

This code solves a linear system
           [     x0 + 2*x1 + 5*x2 + x3 - 4 ]
    F(x) = [ 3*x0 - 4*x1 + 3*x2 - 2*x3 - 7 ],
           [   4*x0 + 3*x1 + 2*x2 - x3 - 1 ]
           [     x0 - 2*x1 - 4*x2 - x3 - 2 ]
which is same as in ksp-basic.c, but with SNES. The solution must be same.
*/

#include <petsc.h>

static const char *const help = "Solve a 4x4 linear system using SNES.\n\n";

static PetscErrorCode FormFunction(SNES, Vec, Vec, void *);
static PetscErrorCode FormJacobian(SNES, Vec, Mat, Mat, void *);

int main(int argc, char *argv[]) {
    Vec x, r;
    Mat J;
    SNES snes;

    /* Initialize PETSc. */
    PetscInitialize(&argc, &argv, NULL, help);

    /* Create vectors. */
    VecCreate(PETSC_COMM_WORLD, &x);
    VecSetSizes(x, PETSC_DECIDE, 4);
    VecSetFromOptions(x);
    VecDuplicate(x, &r);

    /* Create a matrix for the Jacobian. */
    MatCreate(PETSC_COMM_WORLD, &J);
    MatSetSizes(J, PETSC_DECIDE, PETSC_DECIDE, 4, 4);
    MatSetFromOptions(J);
    MatSetUp(J);

    /* Create the SNES solver. */
    SNESCreate(PETSC_COMM_WORLD, &snes);
    SNESSetFunction(snes, r, FormFunction, NULL);
    SNESSetJacobian(snes, J, J, FormJacobian, NULL);
    /* Set SNES as "KSP-only" so that PETSc does not apply complicated methods
       to solve a non-linear equation --- use KSP only to solve this problem. */
    SNESSetType(snes, SNESKSPONLY);
    SNESSetFromOptions(snes);

    /* Set the initial value. */
    VecSet(x, 0);
    /* Solve. */
    SNESSolve(snes, NULL, x);
    /* Print the solution. */
    VecView(x, PETSC_VIEWER_STDOUT_WORLD);

    /* Destroy. */
    VecDestroy(&x);
    VecDestroy(&r);
    MatDestroy(&J);
    SNESDestroy(&snes);

    /* Finalize PETSc. */
    PetscFinalize();
}

static PetscErrorCode FormFunction(SNES snes, Vec x, Vec F, void *ctx) {
    const PetscReal *arrx;
    PetscReal *arrF;

    /* Set values. */
    VecGetArrayRead(x, &arrx);
    VecGetArray(F, &arrF);
    arrF[0] = arrx[0] + 2*arrx[1] + 5*arrx[2] + arrx[3] - 4;
    arrF[1] = 3*arrx[0] - 4*arrx[1] + 3*arrx[2] - 2*arrx[3] - 7;
    arrF[2] = 4*arrx[0] + 3*arrx[1] + 2*arrx[2] - arrx[3] - 1;
    arrF[3] = arrx[0] - 2*arrx[1] - 4*arrx[2] - arrx[3] - 2;
    VecRestoreArrayRead(x, &arrx);
    VecRestoreArray(F, &arrF);

    return 0;
}

static PetscErrorCode FormJacobian(SNES snes, Vec x, Mat J, Mat Jpre,
                                   void *ctx) {
    const PetscReal *arrx;
    PetscReal v[16];
    const PetscInt row[4] = {0, 1, 2, 3}, col[4] = {0, 1, 2, 3};

    /* Set values of Jpre. */
    VecGetArrayRead(x, &arrx);
    v[0] = 1;  v[1] = 2;   v[2] = 5;   v[3] = 1;
    v[4] = 3;  v[5] = -4;  v[6] = 3;   v[7] = -2;
    v[8] = 4;  v[9] = 3;   v[10] = 2;  v[11] = -1;
    v[12] = 1; v[13] = -2; v[14] = -4; v[15] = -1;
    VecRestoreArrayRead(x, &arrx);
    MatSetValues(Jpre, 4, row, 4, col, v, INSERT_VALUES);
    MatAssemblyBegin(Jpre, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(Jpre, MAT_FINAL_ASSEMBLY);

    /* If Jpre and J are different, assemble J also. */
    if (J != Jpre) {
        MatAssemblyBegin(J, MAT_FINAL_ASSEMBLY);
        MatAssemblyEnd(J, MAT_FINAL_ASSEMBLY);
    }

    return 0;
}