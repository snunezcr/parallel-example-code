# Instituto Tecnologico de Costa Rica
# Centro de Investigaciones en Computacion
#
# Asesoria-Practica en Computacion Paralela
# Instituto Costarricense de Electricidad
#
# Julio-Agosto 2011
# 
# Makefile para intergracion por regla trapezoidal

CC=nvcc
CFLAGS=
LDFLAGS=
PREFIX=icevecadd

all: build install

build:
	$(CC) $(CFLAGS) -o $(PREFIX) $(LDFLAGS) $(PREFIX).c

install:
	cp $(PREFIX) ../bin

clean:
	rm -f $(PREFIX)
	rm -f ../bin/$(PREFIX)

