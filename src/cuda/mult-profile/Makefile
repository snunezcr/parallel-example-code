# Instituto Tecnologico de Costa Rica
# Centro de Investigaciones en Computacion
#
# Asesoria-Practica en Computacion Paralela
# Instituto Costarricense de Electricidad
#
# Julio-Agosto 2011
# 
# Makefile para multiplicacion/convolucion con profiling en CUDA C

CC=nvcc
CFLAGS=
LDFLAGS=
PREFIX=icemultprof

all: build install

build:
	$(CC) $(CFLAGS) -o $(PREFIX) $(LDFLAGS) $(PREFIX).cu

install:
	cp $(PREFIX) ../bin

clean:
	rm -f $(PREFIX)
	rm -f ../bin/$(PREFIX)

