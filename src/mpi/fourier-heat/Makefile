# Instituto Tecnologico de Costa Rica
# Centro de Investigaciones en Computacion
#
# Asesoria-Practica en Computacion Paralela
# Instituto Costarricense de Electricidad
#
# Julio-Agosto 2011
# 
# Makefile para aproximacion de difusion de Fourier de temperatura 2D

CC=mpicc
CFLAGS=-g
LDFLAGS=-lm
PREFIX=iceheat

all: build install

build:
	$(CC) $(CFLAGS) -o $(PREFIX) $(LDFLAGS) $(PREFIX).c

install:
	cp $(PREFIX) ../bin

pi:
	make -C pi

dot-product:
	make -C

clean:
	rm -f $(PREFIX)
	rm -f ../bin/$(PREFIX)

