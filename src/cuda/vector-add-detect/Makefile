# Instituto Tecnologico de Costa Rica
# Centro de Investigaciones en Computacion
#
# Asesoria-Practica en Computacion Paralela
# Instituto Costarricense de Electricidad
#
# Julio-Agosto 2011
# 
# Makefile para suma vectorial en CUDA C con deteccion de GPU

CC=nvcc
CFLAGS=
LDFLAGS=
PREFIX=icevecadd_det

all: build install

build:
	$(CC) $(CFLAGS) -o $(PREFIX) $(LDFLAGS) $(PREFIX).cu

install:
	cp $(PREFIX) ../bin

clean:
	rm -f $(PREFIX)
	rm -f ../bin/$(PREFIX)

