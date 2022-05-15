CC = mpicc
CFLAGS = -std=c11 -Wall -Wextra -Wno-unused-parameter
LDLIBS = -lpetsc -lm

snes-jacobian: snes-jacobian.c
	$(CC) $^ -o $@ $(CFLAGS) $(LDLIBS)

snes-linear: snes-linear.c
	$(CC) $^ -o $@ $(CFLAGS) $(LDLIBS)
