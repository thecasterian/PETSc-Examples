CC = mpicc
CFLAGS = -std=c11 -Wall -Wextra -Wno-unused-parameter
LDLIBS = -lpetsc -lm

petsc-basic: petsc-basic.c
	$(CC) $^ -o $@ $(CFLAGS) $(LDLIBS)

ksp-basic: ksp-basic.c
	$(CC) $^ -o $@ $(CFLAGS) $(LDLIBS)

snes-basic: snes-basic.c
	$(CC) $^ -o $@ $(CFLAGS) $(LDLIBS)

snes-jacobian: snes-jacobian.c
	$(CC) $^ -o $@ $(CFLAGS) $(LDLIBS)

dmda-basic: dmda-basic.c
	$(CC) $^ -o $@ $(CFLAGS) $(LDLIBS)

dmda-vector: dmda-vector.c
	$(CC) $^ -o $@ $(CFLAGS) $(LDLIBS)

dmda-poisson-ksp: dmda-poisson-ksp.c
	$(CC) $^ -o $@ $(CFLAGS) $(LDLIBS)
