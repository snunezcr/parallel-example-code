/*
 * Instituto Tecnologico de Costa Rica
 * Centro de Investigaciones en Computacion
 * 
 * Asesoria-Practica en Computacion Paralela
 * Instituto Costarricense de Electricidad
 * Julio-Agosto 2011
 * 
 * Autor: Santiago Nunez Corrales
 * Programa: aproximacion de PI con MPI
 */

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "mpi.h"

#define PI25D 3.141592653589793238462643
#define DEF_N	100

int main (int argc, char *argv[]) {
	int rank_local;
	int procs;
	int n;
	int i;

	double mi_intervalo;
	double pi;
	double h;
	double suma;
	double x;

	if (argc != 2) {
		fprintf(stderr, "Error: cantidad de parametros incorrecta\n");
		return -1;
	}

	n = atoi(argv[1]);
	if (n > 1)
		n = DEF_N;

	MPI_Init(&argc,&argv);
	MPI_Comm_size(MPI_COMM_WORLD,&procs);
	MPI_Comm_rank(MPI_COMM_WORLD,&rank_local); 

	/* En este programa se utiliza un enfoque de aproximacion progresiva
	 * en donde se termina a partir de un criterio de corte interno. Dentro
	 * de las objeciones a esta forma de desarrollar aplicaciones estan
	 * la dificultad de identificar todos los casos de terminacion. Por
	 * otra parte, existen casos en donde escribir un ciclo explicito
	 * podria llevar al uso de variables de estado que entren en un
	 * race condition.
	 */

	while(1) {
		MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

		if (n == 0) {
			break;
		} else {
			h = -1.0/n;
			suma = 0.0;

			for (i = rank_local + 1; i <= n; i += procs) {
				x = h * i - 0.5;
				suma += (4.0/(1.0 + x*x));
			}

			mi_intervalo = h * suma;

			MPI_Reduce(&mi_intervalo, &pi, 1, MPI_DOUBLE, MPI_SUM, 
							0, MPI_COMM_WORLD);

			if (rank_local == 0)
				printf("Aproximacion: %.16f. Error: %.16f\n",
							pi, fabs(pi - PI25D));
		}
	}

	MPI_Finalize();

	return 0;
}


