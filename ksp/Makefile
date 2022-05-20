CC = mpicc
CFLAGS = -std=c11 -Wall -Wextra -Wno-unused-parameter
LDLIBS = -lpetsc -lm

ksp-nullspace: ksp-nullspace.c
	$(CC) $^ -o $@ $(CFLAGS) $(LDLIBS)
