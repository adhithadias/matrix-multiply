# Usage:
# make        # compile all binary
# make clean  # remove ALL binaries and objects

CC = nvcc
CFLAGS = 
INCLUDES = -I../include
LFLAGS = -L../lib
LIBS = -lm

SRCS = $(wildcard *.cu)
OBJS = $(SRCS:.cu=.o)
EXECS = $(SRCS:%.cu=%)

.PHONY: all mm clean

all: ${EXECS}
	@echo  sources compiled

${EXECS}:  %: %.cu
	@echo "creating the executable.."
	$(CC) $(INCLUDES) -o $@ $< $(LFLAGS) $(LIBS)

.cu.ii:
	$(CC) $(INCLUDES) -c $< -o $@ $(LIBS)

clean:
	@echo "removing executables.."
	$(RM) *.o *~ $(EXECS)
