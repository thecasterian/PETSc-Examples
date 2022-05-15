/*
This code creates a 2D scalar field
    v(x, y) = ye^x + 3y^2
on a structured grid within [0, 1] x [0, 1] with DMDAVec and calculates its
gradient using the finite difference method.
*/

#include <petsc.h>

static const char *const help
    = "Create a 2D scalar field and calculate its gradient.\n\n";

int main(int argc, char *argv[]) {
    DM da;
    DMDALocalInfo info;
    PetscReal hx, hy;
    Vec v, gx, gy, gxt, gyt, ex, ey;
    PetscReal **arrv, **arrgx, **arrgy;
    const PetscReal **arrvread;
    PetscReal x, y;
    PetscReal exrel, eyrel, exnorm, eynorm, gxtnorm, gytnorm;
    PetscInt i, j;

    /*----------------------------- Initialize. ------------------------------*/
    /* Initialize PETSc. */
    PetscInitialize(&argc, &argv, NULL, help);

    /*----------------------- Create DMDA and vectors. -----------------------*/
    /* Create the DMDA object. */
    DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,
                 DMDA_STENCIL_STAR, 3, 3, PETSC_DECIDE, PETSC_DECIDE, 1, 1,
                 NULL, NULL, &da);
    DMSetFromOptions(da);
    DMSetUp(da);
    /* Get its information. */
    DMDAGetLocalInfo(da, &info);
    /* Grid spacing in x- and y-directions. */
    hx = 1.0 / (info.mx - 1);
    hy = 1.0 / (info.my - 1);

    /* Create a local vector on the DMDA containing the scalar field. */
    DMCreateLocalVector(da, &v);
    /* For gradients, create global vectors. */
    DMCreateGlobalVector(da, &gx);
    VecDuplicate(gx, &gy);

    /*--------------------------- Fill the vector. ---------------------------*/
    /* Get the internal array of the vector. Since the grid is 2D, the array
       also must be 2D (PetscReal **arrv). */
    DMDAVecGetArray(da, v, &arrv);
    /* Set values. Note that each process has local part of the array only and
       the array is column-major. */
    for (j = info.ys; j < info.ys + info.ym; j++)
        for (i = info.xs; i < info.xs + info.xm; i++) {
            x = i * hx;
            y = j * hy;
            arrv[j][i] = y * PetscExpReal(x) + 3 * y * y;
        }
    DMDAVecRestoreArray(da, v, &arrv);

    /*----------------------- Calculate the gradient. ------------------------*/
    /* Exchange the ghost values. */
    DMLocalToLocalBegin(da, v, INSERT_VALUES, v);
    DMLocalToLocalEnd(da, v, INSERT_VALUES, v);

    /* Get the array of the vectors. Here, a read-only array is used for v. */
    DMDAVecGetArrayRead(da, v, &arrvread);
    DMDAVecGetArray(da, gx, &arrgx);
    DMDAVecGetArray(da, gy, &arrgy);

    /* Calculate gradients. */
    for (j = info.ys; j < info.ys + info.ym; j++)
        for (i = info.xs; i < info.xs + info.xm; i++) {
            /* At i = 0 and i = mx-1, the central differencing scheme is not
               applicable. */
            if (i == 0)
                arrgx[j][i] = (arrvread[j][i+1] - arrvread[j][i]) / hx;
            else if (i == info.mx - 1)
                arrgx[j][i] = (arrvread[j][i] - arrvread[j][i-1]) / hx;
            else
                arrgx[j][i] = (arrvread[j][i+1] - arrvread[j][i-1])
                              / (2 * hx);
            /* Same for j = 0 and j = my-1. */
            if (j == 0)
                arrgy[j][i] = (arrvread[j+1][i] - arrvread[j][i]) / hy;
            else if (j == info.my - 1)
                arrgy[j][i] = (arrvread[j][i] - arrvread[j-1][i]) / hy;
            else
                arrgy[j][i] = (arrvread[j+1][i] - arrvread[j-1][i])
                              / (2 * hy);
        }

    /* Restore vectors. */
    DMDAVecRestoreArrayRead(da, v, &arrvread);
    DMDAVecRestoreArray(da, gx, &arrgx);
    DMDAVecRestoreArray(da, gy, &arrgy);

    /*------------------------- Calculate the error. -------------------------*/
    /* Create vectors for the gradients and the error. */
    DMCreateGlobalVector(da, &gxt);
    VecDuplicate(gxt, &gyt);

    /* Calculate the true gradients value. */
    DMDAVecGetArray(da, gxt, &arrgx);
    DMDAVecGetArray(da, gyt, &arrgy);
    for (j = info.ys; j < info.ys + info.ym; j++)
        for (i = info.xs; i < info.xs + info.xm; i++) {
            x = i * hx;
            y = j * hy;
            arrgx[j][i] = y * PetscExpReal(x);
            arrgy[j][i] = PetscExpReal(x) + 6 * y;
        }
    DMDAVecRestoreArray(da, gxt, &arrgx);
    DMDAVecRestoreArray(da, gyt, &arrgy);

    /* Calculate the error. */
    VecDuplicate(gx, &ex);
    VecDuplicate(gx, &ey);
    VecCopy(gxt, ex);
    VecCopy(gyt, ey);
    VecAXPY(ex, -1, gx);
    VecAXPY(ey, -1, gy);

    /* Calculate the relative error. */
    VecNorm(ex, NORM_1, &exnorm);
    VecNorm(ey, NORM_1, &eynorm);
    VecNorm(gxt, NORM_1, &gxtnorm);
    VecNorm(gyt, NORM_1, &gytnorm);
    exrel = exnorm / gxtnorm;
    eyrel = eynorm / gytnorm;

    PetscPrintf(PETSC_COMM_WORLD, "relative error in x-direction: %f\n", exrel);
    PetscPrintf(PETSC_COMM_WORLD, "relative error in y-direction: %f\n", eyrel);

    /*------------------------------ Finalize. -------------------------------*/
    /* Destroy the DMDA. */
    DMDestroy(&da);

    /* Destroy vectors. */
    VecDestroy(&v);
    VecDestroy(&gx);
    VecDestroy(&gy);
    VecDestroy(&gxt);
    VecDestroy(&gyt);
    VecDestroy(&ex);
    VecDestroy(&ey);

    /* Finalize PETSc. */
    PetscFinalize();
}
