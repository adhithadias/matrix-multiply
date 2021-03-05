# Usage:
# make        # compile all binary
# make clean  # remove ALL binaries and objects

CC = mpicc
CFLAGS = 
INCLUDES = -I../include
LFLAGS = -L../lib
LIBS = -lm -fopenmp

EXECS = omp-mat-mul
SRCS = omp-mat-mul.c
OBJS = $(SRCS:.c=.o)

.PHONY: all mm clean

all: ${EXECS}
	@echo  sources compiled

${EXECS}:  %: %.o
	@echo "creating the executable.."
	$(CC) $(INCLUDES) -o $@ $< $(LFLAGS) $(LIBS)

.c.o:
	$(CC) $(INCLUDES) -c $< -o $@ $(LIBS)

clean:
	@echo "removing executables.."
	$(RM) *.o *~ $(EXECS)
