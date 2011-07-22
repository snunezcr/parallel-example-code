/*
 * Instituto Tecnologico de Costa Rica
 * Centro de Investigaciones en Computacion
 * 
 * Asesoria-Practica en Computacion Paralela
 * Instituto Costarricense de Electricidad
 * Julio-Agosto 2011
 * 
 * Autor: Santiago Nunez Corrales
 * Programa: calculo distribuido del punto producto
 */

#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"

#define VEC_LOCAL_MAX 100

void leer_vector(char *, float *, int, int, int);
float prod_punto_par(float x_local[], float y_local[], int mi_n);
float prod_punto_sec(float *, float *, int n);

int main(int argc, char* argv[]) {
	float x_local[VEC_LOCAL_MAX];
	float y_local[VEC_LOCAL_MAX];
	int n;
	int mi_n;
	float prodpunto;
    	int procs;
	int rank_local;
	int total_lectura;

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &procs);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank_local);

	if (rank_local == 0) {
		printf("Ingrese la cantidad de valores en el vector: ");
		total_lectura = scanf("%d", &n);
	}

	MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

	mi_n = n/procs;

	leer_vector("Vector x: ", x_local, mi_n, procs, rank_local);
	leer_vector("Vector y: ", y_local, mi_n, procs, rank_local);

	prodpunto = prod_punto_par(x_local, y_local, mi_n);

	if (rank_local == 0)
		printf("El valor del producto punto es: %f\n", prodpunto);

	MPI_Finalize();
	return 0;
}
	   
void leer_vector(char* entrada, float *v_local, int mi_n, int procs,
							int rank_local) {
	int i;
	int q;
	int total_lectura;
    	float temp[VEC_LOCAL_MAX];
    	MPI_Status status;

	if (rank_local == 0) {
		printf("Datos: %s", entrada);
		
		for (i = 0; i < mi_n; i++)
		    total_lectura = scanf("%f", &v_local[i]);

		for (q = 1; q < procs; q++) {
			for (i = 0; i < mi_n; i++)
				total_lectura = scanf("%f", &temp[i]);
		
			MPI_Send(temp, mi_n, MPI_FLOAT, q, 0, MPI_COMM_WORLD);
		}
	} else {
		MPI_Recv(v_local, mi_n, MPI_FLOAT, 0, 0, MPI_COMM_WORLD,
		    						&status);
	}
}
 
float prod_punto_sec(float *x, float *y, int n) {
	int i; 
	float suma;

	suma = 0.0;

	for (i = 0; i < n; i++)
		suma = suma + x[i]*y[i];

	return suma;
}

float prod_punto_par(float  *x_local, float  *y_local, int  mi_n) {
	float local_prodpunto;
	float prodpunto;

	prodpunto = 0;
	local_prodpunto = prod_punto_sec(x_local, y_local, mi_n);

	MPI_Reduce(&local_prodpunto, &prodpunto, 1, MPI_FLOAT,
        MPI_SUM, 0, MPI_COMM_WORLD);

	return prodpunto;
}


