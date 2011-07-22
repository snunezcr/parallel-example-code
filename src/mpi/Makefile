# Instituto Tecnologico de Costa Rica
# Centro de Investigaciones en Computacion
#
# Asesoria-Practica en Computacion Paralela
# Instituto Costarricense de Electricidad
#
# Julio-Agosto 2011
# 
# Makefile para codigo MPI

CC=mpicc
CFLAGS=-g
LDFLAGS=-lm

all: hello integration pi dotprod vecmat pimc heat

hello:
	make -C hello-world

integration:
	make -C integral

pi:
	make -C pi-calc

dotprod:
	make -C dot-product

vecmat:
	make -C vector-matrix

pimc:
	make -C pi-montecarlo

heat:
	make -C fourier-heat

clean:
	make -C hello-world clean
	make -C integral clean
	make -C pi-calc clean
	make -C dot-product clean
	make -C vector-matrix clean
	make -C pi-montecarlo clean
	make -C fourier-heat clean

