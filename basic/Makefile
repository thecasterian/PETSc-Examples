CC = mpicc
CFLAGS = -std=c11 -Wall -Wextra -Wno-unused-parameter
LDLIBS = -lpetsc -lm

petsc-basic: petsc-basic.c
	$(CC) $^ -o $@ $(CFLAGS) $(LDLIBS)

ksp-basic: ksp-basic.c
	$(CC) $^ -o $@ $(CFLAGS) $(LDLIBS)

snes-basic: snes-basic.c
	$(CC) $^ -o $@ $(CFLAGS) $(LDLIBS)