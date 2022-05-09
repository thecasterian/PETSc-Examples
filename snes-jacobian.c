/*
This code solves the same problem with snes-basic.c, but with the Jacobian of
F(x):
           [ 2*x0 2*x1 -4 ]
    J(x) = [    1   -1 -1 ]
           [    3   x2 x1 ]
The solution must be same.
*/

#include <petsc.h>

static const char *const help
    = "Solve a non-linear equation using SNES with the Jacobian.\n\n";

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
    VecSetSizes(x, PETSC_DECIDE, 3);
    VecSetFromOptions(x);
    VecDuplicate(x, &r);

    /* Create a matrix for the Jacobian. */
    MatCreate(PETSC_COMM_WORLD, &J);
    MatSetSizes(J, PETSC_DECIDE, PETSC_DECIDE, 3, 3);
    MatSetFromOptions(J);
    MatSetUp(J);

    /* Create the SNES solver. */
    SNESCreate(PETSC_COMM_WORLD, &snes);
    SNESSetFunction(snes, r, FormFunction, NULL);
    /* Specify the Jacobian. The second parameter is the Jacobian matrix and the
       third is a matrix used to construct a preconditioner. In most cases these
       are same. */
    SNESSetJacobian(snes, J, J, FormJacobian, NULL);
    SNESSetFromOptions(snes);

    /* Set the initial value. */
    VecSet(x, 1);
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
    arrF[0] = arrx[0] * arrx[0] + arrx[1] * arrx[1] - 4 * arrx[2] - 4;
    arrF[1] = arrx[0] - arrx[1] - arrx[2] - 3;
    arrF[2] = 3 * arrx[0] + arrx[1] * arrx[2] - 4;
    VecRestoreArrayRead(x, &arrx);
    VecRestoreArray(F, &arrF);

    return 0;
}

static PetscErrorCode FormJacobian(SNES snes, Vec x, Mat J, Mat Jpre,
                                   void *ctx) {
    /* Similar to SNESSetJacobian(), J is the Jacobian Matrix and Jpre is a
       matrix used to construct a preconditioner. Since two matrices are same
       in SNESSetJacobian(), setting only one matrix is sufficient. However,
       it is usual to set Jpre in this case since under some options J may be a
       numerically approximated Jacobian calculated by PETSc so need not be
       set manually. */
    const PetscReal *arrx;
    PetscReal v[9];
    const PetscInt row[3] = {0, 1, 2}, col[3] = {0, 1, 2};

    /* Set values of Jpre. */
    VecGetArrayRead(x, &arrx);
    v[0] = 2 * arrx[0]; v[1] = 2 * arrx[1]; v[2] = -4;
    v[3] = 1;           v[4] = -1;          v[5] = -1;
    v[6] = 3;           v[7] = arrx[2];     v[8] = arrx[1];
    VecRestoreArrayRead(x, &arrx);
    MatSetValues(Jpre, 3, row, 3, col, v, INSERT_VALUES);
    MatAssemblyBegin(Jpre, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(Jpre, MAT_FINAL_ASSEMBLY);

    /* If Jpre and J are different, assemble J also. */
    if (J != Jpre) {
        MatAssemblyBegin(J, MAT_FINAL_ASSEMBLY);
        MatAssemblyEnd(J, MAT_FINAL_ASSEMBLY);
    }

    return 0;
}
