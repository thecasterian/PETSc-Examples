/*
This code solves the same poisson equation as in dmda-poisson-ksp.c but with
SNES.

SNES supports geometric multigrid method on DMDA. Enable it with `-pc_type mg`
and `-da_refine N` command line option where N defines the number of levels in
multigrid method.
*/

#include <petsc.h>

const char *const help
    = "Solve a 2D Poisson equation on a unit square with SNES.\n\n";

PetscErrorCode FormFunctionLocal(DMDALocalInfo *, void *, void *, void *);
PetscErrorCode FormJacobianLocal(DMDALocalInfo *, void *, Mat, Mat, void *);
PetscErrorCode FormInitial(DM, Vec);
PetscErrorCode FormExactSolution(DM, Vec);

int main(int argc, char *argv[]) {
    DM da;
    Vec u, ut;
    SNES snes;
    PetscReal e;

    /* Initialize PETSc. */
    PetscInitialize(&argc, &argv, NULL, help);

    /* Create DMDA. */
    DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,
                 DMDA_STENCIL_STAR, 3, 3, PETSC_DECIDE, PETSC_DECIDE, 1, 1,
                 NULL, NULL, &da);
    DMSetFromOptions(da);
    DMSetUp(da);

    /* Create SNES. */
    SNESCreate(PETSC_COMM_WORLD, &snes);
    /* Set DMDA for SNES. */
    SNESSetDM(snes, da);
    /* DMDASNESSetFunctionLocal() is more convenient variant of
       SNESSetFunction() for DMDA+SNES. See the definition of
       FormFunctionLocal() for more information. */
    DMDASNESSetFunctionLocal(da, INSERT_VALUES, FormFunctionLocal, NULL);
    /* Similar variant of SNESSetJacobian() for DMDA+SNES. */
    DMDASNESSetJacobianLocal(da, FormJacobianLocal, NULL);
    /* This is a linear problem; set to KSP-only. */
    SNESSetType(snes, SNESKSPONLY);
    SNESSetFromOptions(snes);

    /* Create a "temporary" vector for the initial value. */
    DMGetGlobalVector(da, &u);
    /* Set the initial value. */
    FormInitial(da, u);
    /* Solve SNES. */
    SNESSolve(snes, NULL, u);

    /* Unlike DMDA+KSP, DMDA+SNES has the special feature named "grid
       sequencing" which gradually refine the grid to get more accurate solution
       with high resolution. Therefore, the DMDA set for SNES before may be
       different from a DMDA after solve so the DMDA and the solution vector
       must be get again. */
    /* Restore the temporary vector after the use. */
    DMRestoreGlobalVector(da, &u);
    /* Destroy the old DMDA. */
    DMDestroy(&da);
    /* Get the new DMDA from SNES. */
    SNESGetDM(snes, &da);
    /* Get the solution vector which fits into the new DMDA. */
    SNESGetSolution(snes, &u);
    /* Note that the new DMDA and the solution vector is managed by SNES
       automatically. Do not destroy it manually. */

    /* Calculate the maximum error. */
    DMCreateGlobalVector(da, &ut);
    FormExactSolution(da, ut);
    VecAXPY(ut, -1.0, u);
    VecNorm(ut, NORM_INFINITY, &e);
    PetscPrintf(PETSC_COMM_WORLD, "Maximum error: %g\n", e);

    /* Clean up. */
    SNESDestroy(&snes);
    VecDestroy(&ut);

    /* Finalize PETSc. */
    PetscFinalize();
}

PetscErrorCode FormFunctionLocal(DMDALocalInfo *info, void *_arrx, void *_arrF,
                                 void *ctx) {
    /* FormFunction() in traditional SNES routine, such as in snes-basic.c,
       accpets x and F as a vector. In contrast, since DMDA is used here, this
       function accepts x and F as a 2D array to make the code simple. */
    PetscReal **arrx = _arrx, **arrF = _arrF;
    PetscReal hx, hy;
    PetscReal x, y, f;
    PetscInt i, j;

    hx = 1.0 / (info->mx - 1);
    hy = 1.0 / (info->my - 1);

    for (j = info->ys; j < info->ys + info->ym; j++)
        for (i = info->xs; i < info->xs + info->xm; i++) {
            x = i * hx;
            y = j * hy;

            if (i == 0)
                arrF[j][i] = arrx[j][i] - PetscCosReal(2*PETSC_PI*y);
            else if (i == info->mx - 1)
                arrF[j][i] = arrx[j][i] - PetscCosReal(2*PETSC_PI*y);
            else if (j == 0)
                arrF[j][i] = arrx[j][i] - PetscCosReal(2*PETSC_PI*x);
            else if (j == info->my - 1)
                arrF[j][i] = arrx[j][i] - PetscCosReal(2*PETSC_PI*x);
            else {
                f = 8.0*PETSC_PI*PETSC_PI * PetscCosReal(2*PETSC_PI*x)
                    * PetscCosReal(2*PETSC_PI*y);
                arrF[j][i]
                    = hy/hx * (2*arrx[j][i] - arrx[j][i+1] - arrx[j][i-1])
                      + hx/hy * (2*arrx[j][i] - arrx[j+1][i] - arrx[j-1][i])
                      - hx*hy * f;
            }
        }

    return 0;
}

PetscErrorCode FormJacobianLocal(DMDALocalInfo *info, void *_arrx, Mat J,
                                 Mat Jpre, void *ctx) {
    /* Set the Jacobian matrix in the similar manner of FormMatrix() in
       dmda-poisson-ksp.c. Since the equation is linear, values in the matrix
       are totally identical. */
    PetscReal **arrx = _arrx;
    PetscReal hx, hy;
    MatStencil row, col[5];
    PetscReal v[5];
    PetscInt ncols;
    PetscInt i, j;

    hx = 1.0 / (info->mx - 1);
    hy = 1.0 / (info->my - 1);

    for (j = info->ys; j < info->ys + info->ym; j++)
        for (i = info->xs; i < info->xs + info->xm; i++) {
            row.i = i;
            row.j = j;
            col[0].i = i;
            col[0].j = j;

            if (i == 0 || i == info->mx - 1 || j == 0 || j == info->my - 1) {
                v[0] = 1.0;
                ncols = 1;
            } else {
                v[0] = 2.0 * (hy / hx + hx / hy);
                col[1].i = i + 1;
                col[1].j = j;
                v[1] = -hy / hx;
                col[2].i = i - 1;
                col[2].j = j;
                v[2] = -hy / hx;
                col[3].i = i;
                col[3].j = j + 1;
                v[3] = -hx / hy;
                col[4].i = i;
                col[4].j = j - 1;
                v[4] = -hx / hy;
                ncols = 5;
            }

            MatSetValuesStencil(Jpre, 1, &row, ncols, col, v, INSERT_VALUES);
        }

    MatAssemblyBegin(Jpre, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(Jpre, MAT_FINAL_ASSEMBLY);
    if (J != Jpre) {
        MatAssemblyBegin(J, MAT_FINAL_ASSEMBLY);
        MatAssemblyEnd(J, MAT_FINAL_ASSEMBLY);
    }

    return 0;
}

PetscErrorCode FormInitial(DM da, Vec u) {
    DMDALocalInfo info;
    PetscReal hx, hy;
    PetscReal **arru, x, y;
    PetscInt i, j;

    DMDAGetLocalInfo(da, &info);
    hx = 1.0 / (info.mx - 1);
    hy = 1.0 / (info.my - 1);

    DMDAVecGetArray(da, u, &arru);
    for (j = info.ys; j < info.ys + info.ym; j++)
        for (i = info.xs; i < info.xs + info.xm; i++) {
            x = i * hx;
            y = j * hy;

            /* The initial value equals to the boundary condition on the
               boundary and equals to 0 elsewhere. */
            if (i == 0)
                arru[j][i] = PetscCosReal(2*PETSC_PI*y);
            else if (i == info.mx - 1)
                arru[j][i] = PetscCosReal(2*PETSC_PI*y);
            else if (j == 0)
                arru[j][i] = PetscCosReal(2*PETSC_PI*x);
            else if (j == info.my - 1)
                arru[j][i] = PetscCosReal(2*PETSC_PI*x);
            else
                arru[j][i] = 0.0;
        }

    return 0;
}

PetscErrorCode FormExactSolution(DM da, Vec ut) {
    DMDALocalInfo info;
    PetscReal hx, hy;
    PetscReal **arrut, x, y;
    PetscInt i, j;

    DMDAGetLocalInfo(da, &info);
    hx = 1.0 / (info.mx - 1);
    hy = 1.0 / (info.my - 1);

    DMDAVecGetArray(da, ut, &arrut);
    for (j = info.ys; j < info.ys + info.ym; j++)
        for (i = info.xs; i < info.xs + info.xm; i++) {
            x = i * hx;
            y = j * hy;
            arrut[j][i] = PetscCosReal(2*PETSC_PI*x)
                          * PetscCosReal(2*PETSC_PI*y);
        }
    DMDAVecRestoreArray(da, ut, &arrut);

    return 0;
}
