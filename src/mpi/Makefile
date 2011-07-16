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

all: hello integration

hello:
	make -C hello-world

integration:
	make -C integral

#pi:
#	make -C pi

#dot-product:
#	make -C dot-product

#sparse-matrix:
#	make -C sparse-matrix

#fox-matrix:
#	make -C fox-matrix

#ring-comm:
#	make -C ring-comm

#jacobi:
#	make -C jacobi

#simple-sort:
#	make -C simple-sort

#bitonic-sort:
#	make -C bitonic-sort

#poisson:
#	make -C poisson

clean:
	make -C hello-world clean
	make -C integral clean
#	make -C pi clean
#	make -C dot-product clean
#	make -C sparse-matrix clean
#	make -C fox-matrix clean
#	make -C ring-comm clean
#	make -C jacobi clean
#	make -C simple-sort clean
#	make -C bitonic-sort clean
#	make -C poisson clean

