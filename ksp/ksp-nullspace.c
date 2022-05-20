/*
If the matrix in a linear system is singular then the system may have infinitely
many solutions. Consider a linear system Ax = b where A is a 4x4 matrix
  [  2  1  4 -1 ]
  [  3 -2  1  0 ]
  [  5  1 -3  2 ]
  [ -1  3  3 -1 ]
and b is a 4x1 column vector
  [ 10 ]
  [ 13 ]
  [  1 ]
  [ -3 ].
The answer has the following form:
  [ -13 ]     [  3 ]
  [  -6 ]     [ -2 ]
  [  27 ] t + [  0 ]
  [  76 ]     [ -6 ]
where t is a real number.

In such case, KSP may fail to converge and find the solution. Fortunately, KSP
can solve a singular linear system by setting the null space of A. Here, the
null space is defined by one basis vector [ -13 -6 27 76 ]^T.
*/

#include <petsc.h>

static const char *const help
    = "Solve a 4x4 singular linear system removing its null space.\n\n";

/* Matrix A in array. */
PetscReal arrA[4][4] = {
    { 2,  1,  4, -1},
    { 3, -2,  1,  0},
    { 5,  1, -3,  2},
    {-1,  3,  3, -1},
};
/* Vector b in array. */
PetscReal arrb[4] = {
    10,
    13,
    1,
    -3,
};
/* Basis of the null space of A in array. */
PetscReal arrbasis[4] = {
    -13,
    -6,
    27,
    76,
};

int main(int argc, char *argv[]) {
    Mat A;
    Vec b, x;
    KSP ksp;
    PC pc;
    Vec basis;
    MatNullSpace ns;
    PetscReal norm;
    PetscInt i, j[4] = {0, 1, 2, 3};

    /* Initialize PETSc. */
    PetscInitialize(&argc, &argv, NULL, help);

    /* Create the matrix A. */
    MatCreate(PETSC_COMM_WORLD, &A);
    MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, 4, 4);
    MatSetFromOptions(A);
    MatSetUp(A);

    /* Set A. */
    for (i = 0; i < 4; i++)
        MatSetValues(A, 1, &i, 4, j, arrA[i], INSERT_VALUES);
    MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);

    /* Create the vector b. */
    VecCreate(PETSC_COMM_WORLD, &b);
    VecSetSizes(b, PETSC_DECIDE, 4);
    VecSetFromOptions(b);

    /* Set b. */
    VecSetValues(b, 4, j, arrb, INSERT_VALUES);
    VecAssemblyBegin(b);
    VecAssemblyEnd(b);

    /* Create the basis of the null space of A. */
    VecDuplicate(b, &basis);
    VecSetValues(basis, 4, j, arrbasis, INSERT_VALUES);
    VecAssemblyBegin(basis);
    VecAssemblyEnd(basis);
    /* PETSc requires that the basis must have L2 norm of 1. Normalize it. */
    VecNorm(basis, NORM_2, &norm);
    VecScale(basis, 1.0 / norm);

    /* Create the null space. */
    MatNullSpaceCreate(
        PETSC_COMM_WORLD,   /* MPI commonicator. Must be same with A. */
        PETSC_FALSE,        /* True if there are constant vector (a vector with
                               all identical values) among bases. This special
                               case is handled separately because it is very
                               common. */
        1,                  /* Number of bases excluding the constant vector. */
        &basis,             /* Array of bases excluding the constant vector. If
                               the only basis is a constant vector, set this to
                               NULL. */
        &ns
    );
    /* Set the null space. */
    MatSetNullSpace(A, ns);

    /* Create KSP. Here, the type of preconditioner is explicitly specified as
       `PC_NONE` (no preconditioner) since the default preconditioner, ILU, must
       fails on a singular matrix. */
    KSPCreate(PETSC_COMM_WORLD, &ksp);
    KSPSetOperators(ksp, A, A);
    KSPGetPC(ksp, &pc);
    PCSetType(pc, PCNONE);
    KSPSetFromOptions(ksp);

    /* Solve KSP. */
    VecDuplicate(b, &x);
    KSPSolve(ksp, b, x);
    VecView(x, PETSC_VIEWER_STDOUT_WORLD);

    /* Destroy. */
    MatDestroy(&A);
    VecDestroy(&b);
    VecDestroy(&basis);
    MatNullSpaceDestroy(&ns);
    VecDestroy(&x);
    KSPDestroy(&ksp);

    /* Finalize PETSc. */
    PetscFinalize();
}
