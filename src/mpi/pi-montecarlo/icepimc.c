/*
 * Instituto Tecnologico de Costa Rica
 * Centro de Investigaciones en Computacion
 * 
 * Asesoria-Practica en Computacion Paralela
 * Instituto Costarricense de Electricidad
 * Julio-Agosto 2011
 * 
 * Autor: Santiago Nunez Corrales
 * Programa: calculo de Pi usando un algoritmo de Monte Carlo
 */

#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"

#define DEF_MUESTRAS 1000
#define DEF_RONDAS 10
#define sqr(x) ((x)*(x))

double muestrear(int);

int main (int argc, char *argv[]) {
	double	mi_pi;
	double suma_pi;
	double pi;
	double pi_prom;
	int rank_local;
	int procs;
	int muestras;
	int rondas;
	int ret;
	int i;

	if (argc != 3) {
		fprintf(stderr, "Error de parametros.\n");
		return -1;
	}
	
	muestras = atoi(argv[1]);
	if (muestras < 1)
		muestras = DEF_MUESTRAS;

	rondas = atoi(argv[2]);
	if (rondas < 1)
		rondas = DEF_RONDAS;

	MPI_Init(&argc,&argv);
	MPI_Comm_size(MPI_COMM_WORLD,&procs);
	MPI_Comm_rank(MPI_COMM_WORLD,&rank_local);

	if (rank_local == 0) 
		printf ("Se usan %d para aproximar pi (3.1415926535)\n", procs);

	srandom (rank_local);
	pi_prom = 0;

	for (i = 0; i < rondas; i++) {
		mi_pi = muestrear(muestras);

		ret = MPI_Reduce(&mi_pi, &suma_pi, 1, MPI_DOUBLE, MPI_SUM,
							0, MPI_COMM_WORLD);
		if (ret != MPI_SUCCESS)
			printf("%d: Error en MPI_Reduce\n", rank_local);

		if (rank_local == 0) {
			pi = suma_pi/procs;
			pi_prom = ((pi_prom * i) + pi)/(i + 1); 
      			printf("Lanzamientos: %8d\tpi promedio: %10.8f\n",
            					(muestras * (i + 1)), pi_prom);
		}
	}

	MPI_Finalize();
	return 0;
}

double muestrear(int muestras) {
	double val_x;
	double val_y;
	double pi;
	double r; 
  	int aciertos;
	int n;
  	unsigned int periodo;

	if (sizeof(periodo) != 4) {
    		fprintf(stderr, "Esta arquitectura tiene un int menor a\n");
    		fprintf(stderr, "4 bytes. Abortando.\n");
		exit(1);
	}

	periodo = 2 << (31 - 1);
	aciertos = 0;

	for (n = 1; n <= muestras; n++) {
		r = (double)random()/periodo;
		val_x = (2.0 * r) - 1.0;
		r = (double)random()/periodo;
		val_y = (2.0 * r) - 1.0;

		if ((sqr(val_x) + sqr(val_y)) <= 1.0)
			aciertos++;
	}

	pi = 4.0 * (double)aciertos/(double)muestras;
	return pi ;
}

