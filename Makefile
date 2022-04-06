CC = mpicc
CFLAGS = -std=c11 -Wall -Wextra -Wno-unused-parameter
LDLIBS = -lpetsc -lm

ksp-basic: ksp-basic.c
	$(CC) $^ -o $@ $(CFLAGS) $(LDLIBS)

snes-basic: snes-basic.c
	$(CC) $^ -o $@ $(CFLAGS) $(LDLIBS)
