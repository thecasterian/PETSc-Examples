/*
This code solves a linear system Ax = b using KSP where A is a 4x4 matrix
  [  1  2  5  1 ]
  [  3 -4  3 -2 ]
  [  4  3  2 -1 ]
  [  1 -2 -4 -1 ]
and b is a 4x1 column vector
  [  4 ]
  [  7 ]
  [  1 ]
  [  2 ].
Note that the answer x is
  [  3 ]
  [ -2 ]
  [  0 ]
  [  5 ].
*/

#include <petsc.h>

static const char *const help = "Solve a 4x4 linear system using KSP.\n\n";

/* Matrix A in array. */
PetscReal arrA[4][4] = {
    { 1,  2,  5,  1},
    { 3, -4,  3, -2},
    { 4,  3,  2, -1},
    { 1, -2, -4, -1},
};
/* Vector b in array. */
PetscReal arrb[4] = {
    4,
    7,
    1,
    2,
};

int main(int argc, char *argv[]) {
    Mat A;
    Vec b, x;
    KSP ksp;
    PetscInt i, j[4] = {0, 1, 2, 3};

    /*----------------------------- Initialize. ------------------------------*/
    /* Initialize PETSc. */
    PetscInitialize(&argc, &argv, NULL, help);

    /*---------------------------- Build matrix. -----------------------------*/
    /* Create the matrix A. */
    MatCreate(PETSC_COMM_WORLD, &A);
    /* Set the matrix size. The second and third parameter are the local size
       (the number of rows and columns) and the fourth and fifth are the global
       size. Here, let PETSc decide the local size automatically with
       PETSC_DECIDE. */
    MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, 4, 4);
    /* Determine other matrix properties from command-line options. */
    MatSetFromOptions(A);
    /* Set up the internal matrix data structures. */
    MatSetUp(A);
    /* Set the elements row by row. If ADD_VALUES is used instead of
       INSERT_VALUES, the given values will be added to already existing
       elements. */
    for (i = 0; i < 4; i++)
        MatSetValues(A, 1, &i, 4, j, arrA[i], INSERT_VALUES);
    /* Assemble the matrix; now it is ready to use. MatAssemblyBegin() and
       MatAssemblyEnd() must be paired one-to-one. */
    MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);

    /*---------------------------- Build vector. -----------------------------*/
    /* Create the vector b. */
    VecCreate(PETSC_COMM_WORLD, &b);
    /* Set the vector size. Again, let PETSc decide the local size
       automatically. */
    VecSetSizes(b, PETSC_DECIDE, 4);
    /* Determine other vector properties from command-line options. */
    VecSetFromOptions(b);
    /* For a basic use of the Vec class, it is not mandatory to call VecSetUp()
       explicitly since set-up will happen automatically. */
    /* Set the elements. */
    VecSetValues(b, 4, j, arrb, INSERT_VALUES);
    /* Assemble the vector. */
    VecAssemblyBegin(b);
    VecAssemblyEnd(b);

    /*------------------------------ Build KSP. ------------------------------*/
    /* Create the KSP solver. */
    KSPCreate(PETSC_COMM_WORLD, &ksp);
    /* Set the matrix associated with the linear system to solve. The second
       parameter indicates the matrix that defines the system and the third
       is the matrix used to construct a preconditioner. These two parameters
       are usually same. */
    KSPSetOperators(ksp, A, A);
    /* Determine other KSP properties from command-line options. */
    KSPSetFromOptions(ksp);

    /*-------------------------------- Solve. --------------------------------*/
    /* Create the solution vector x duplicating from the vector b. */
    VecDuplicate(b, &x);
    /* Solve the linear system. */
    KSPSolve(ksp, b, x);
    /* Print the solution to stdout. */
    VecView(x, PETSC_VIEWER_STDOUT_WORLD);

    /*------------------------------ Finalize. -------------------------------*/
    /* Destroy the matrix, vectors, and KSP. */
    MatDestroy(&A);
    VecDestroy(&b);
    VecDestroy(&x);
    KSPDestroy(&ksp);

    /* Finalize PETSc. */
    PetscFinalize();
}
