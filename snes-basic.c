/*
This code solves a non-linear equation F(x) = 0 using SNES where
         [ x0^2 + x1^2 - 4*x2 - 4 ]
  F(x) = [       x0 - x1 - x2 - 3 ].
         [       3*x0 + x1*x2 - 4 ]
The solution is
  [  2 ]
  [ -2 ].
  [  1 ]
*/

#include <petsc.h>

static const char *const help = "Solve a non-linear equation using SNES.\n\n";

static PetscErrorCode FormFunction(SNES, Vec, Vec, void *);

int main(int argc, char *argv[]) {
    Vec x, r;
    SNES snes;

    /*----------------------------- Initialize. ------------------------------*/
    PetscInitialize(&argc, &argv, NULL, help);

    /*---------------------------- Build vectors. ----------------------------*/
    /* Create the solution vector x. */
    VecCreate(PETSC_COMM_WORLD, &x);
    /* Set the vector size. */
    VecSetSizes(x, PETSC_DECIDE, 3);
    /* Determine other vector properties from command-line options. */
    VecSetFromOptions(x);
    /* Set the initial values: all zeros. */
    VecSet(x, 0);
    /* Create the residual vector r duplicating the solution vector. */
    VecDuplicate(x, &r);

    /*---------------------------- Build SNES. -------------------------------*/
    /* Create the SNES solver. */
    SNESCreate(PETSC_COMM_WORLD, &snes);
    /* Set the function to be solved. The last parameter is a user-supplied
       data sended to the function. It is not used here so set to NULL. */
    SNESSetFunction(snes, r, FormFunction, NULL);
    /* Determine other SNES properties from command-line options. */
    SNESSetFromOptions(snes);

    /*----------------------------- Solve. -----------------------------------*/
    /* Solve the non-linear equation. The second parameter is the RHS of the
       equation; set to NULL if 0. */
    SNESSolve(snes, NULL, x);
    /* Print the solution to stdout. */
    VecView(x, PETSC_VIEWER_STDOUT_WORLD);

    /*----------------------------- Finalize. --------------------------------*/
    /* Destroy the vectors and SNES. */
    VecDestroy(&x);
    VecDestroy(&r);
    SNESDestroy(&snes);

    /* Finalize PETSc. */
    PetscFinalize();
}

/* This function calculates the vector F(x) for the given vector x. The last
   parameter ctx is the pointer to the user-supplied data, such as coefficients
   in terms. */
static PetscErrorCode FormFunction(SNES snes, Vec x, Vec F, void *ctx) {
    const PetscReal *arrx;
    PetscReal *arrF;

    /* Get pointers to the internal array of elements of the vectors x and F(x).
       `Read` means the resultant pointer is read-only (pointer to const). */
    VecGetArrayRead(x, &arrx);
    VecGetArray(F, &arrF);

    /* Calculate F(x). */
    arrF[0] = arrx[0] * arrx[0] + arrx[1] * arrx[1] - 4 * arrx[2] - 4;
    arrF[1] = arrx[0] - arrx[1] - arrx[2] - 3;
    arrF[2] = 3 * arrx[0] + arrx[1] * arrx[2] - 4;

    /* Tell PETSc that pointers are no longer used. These calls set pointers
       to NULL. VecGetArray() and VecRestoreArray() must be pair one-to-one.
       Same for VecGetArrayRead() and VecRestoreArrayRead(). */
    VecRestoreArrayRead(x, &arrx);
    VecRestoreArray(F, &arrF);
    /* Now arrx and arrF are null pointers so that any accidental access to
       the internal array of elements can be blocked. */

    /* No error; return 0. */
    return 0;
}
