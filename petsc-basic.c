/*
Basic structure of a PETSc program.
*/

#include <petsc.h>

const char *const help = "Basic PETSc program\n";

int main(int argc, char *argv[]) {
    PetscErrorCode ierr;
    PetscInt n = 0;
    PetscReal r = 1.5;

    int rank;
    int size;

    /* Initialize PETSc. The first and second parameter are pointers to `argc`
       and `argv`, respectively. The third is the path to the option file. The
       last is a help string, which is printed when the program is executed with
       the option `-help`. */
    ierr = PetscInitialize(&argc, &argv, NULL, help);
    /* Check the return value and quit if an error occurred. Error check is not
       mandatory. */
    CHKERRQ(ierr);

    /* User can define a custom option. */
    PetscOptionsGetInt(NULL, NULL, "-n", &n, NULL);
    PetscOptionsGetReal(NULL, NULL, "-r", &r, NULL);

    /* PETSc initializes MPI automatically. Any MPI function can be used. PETSc
       defines the communicator `PETSC_COMM_WORLD`, which is the set of all
       processes that participate to PETSc. */
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    MPI_Comm_size(PETSC_COMM_WORLD, &size);

    /* PetscPrintf() prints to the standard output, only from the first
       process. This is useful to print an information same in all processes. */
    PetscPrintf(PETSC_COMM_WORLD, "n = %d, r = %lf\n", n, r);
    /* PetscSynchronizedPrintf() prints to the standard output in the order of
       rank. Therefore, output of the first processor precedes that of the
       second and so on. */
    PetscSynchronizedPrintf(PETSC_COMM_WORLD, "rank = %d, size = %d\n", rank,
                            size);
    /* PetscSynchronizedFlush() flushes a file such as the standard input so
       that every synchronized output comes before the program exits. */
    PetscSynchronizedFlush(PETSC_COMM_WORLD, PETSC_STDOUT);

    /* Finalize PETSc. */
    PetscFinalize();
}
