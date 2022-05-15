CC = mpicc
CFLAGS = -std=c11 -Wall -Wextra -Wno-unused-parameter
LDLIBS = -lpetsc -lm

dmda-basic: dmda-basic.c
	$(CC) $^ -o $@ $(CFLAGS) $(LDLIBS)

dmda-vector: dmda-vector.c
	$(CC) $^ -o $@ $(CFLAGS) $(LDLIBS)

dmda-poisson-ksp: dmda-poisson-ksp.c
	$(CC) $^ -o $@ $(CFLAGS) $(LDLIBS)

dmda-poisson-snes: dmda-poisson-snes.c
	$(CC) $^ -o $@ $(CFLAGS) $(LDLIBS)
