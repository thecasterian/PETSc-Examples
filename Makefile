CC = mpicc
CFLAGS = -std=c11 -Wall -Wextra
LDLIBS = -lpetsc

ksp-basic: ksp-basic.c
	$(CC) $^ -o $@ $(CFLAGS) $(LDLIBS)
