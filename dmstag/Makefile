CC = mpicc
CFLAGS = -std=c11 -Wall -Wextra -Wno-unused-parameter
LDLIBS = -lpetsc -lm

dmstag-basic: dmstag-basic.c
	$(CC) $^ -o $@ $(CFLAGS) $(LDLIBS)