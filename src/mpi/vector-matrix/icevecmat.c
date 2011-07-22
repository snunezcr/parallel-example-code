/*
 * Instituto Tecnologico de Costa Rica
 * Centro de Investigaciones en Computacion
 * 
 * Asesoria-Practica en Computacion Paralela
 * Instituto Costarricense de Electricidad
 * Julio-Agosto 2011
 * 
 * Autor: Santiago Nunez Corrales
 * Programa: multiplicacion de una matriz por un vector con
 * 	     operaciones colectivas.
 */

#include <stdio.h>
#include "mpi.h"

#define MAX_N 100

typedef float MAT_LOCAL_T[MAX_N][MAX_N];

void leer_matriz(char *, MAT_LOCAL_T, int, int, int, int);
void leer_vector(char *, float *, int, int, int);
void mat_vec_par(MAT_LOCAL_T, int, int, float *, float *, float *, int, int);
void imprimir_mat(char *, MAT_LOCAL_T, int, int, int, int);
void imprimir_vec(char *, float *, int, int, int);

int main(int argc, char* argv[]) {
	int rank_local;
	int procs;
	MAT_LOCAL_T a_local; 
	float global_x[MAX_N];
	float x_local[MAX_N];
	float y_local[MAX_N];
	int m;
	int n;
	int m_local;
	int n_local;
	int total_leidos;

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &procs);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank_local);

	if (rank_local == 0) {
		printf("Ingrese el orden de la matriz (m n): ");
		total_leidos = scanf("%d %d", &m, &n);
	}

	MPI_Bcast(&m, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

	m_local = m/procs;
	n_local = n/procs;

	leer_matriz("Ingrese la matriz: ", a_local, m_local, n, rank_local,
									procs);
	imprimir_mat("Matriz leida: ", a_local, m_local, n, rank_local, procs);
	leer_vector("Ingrese el vector: ", x_local, n_local, rank_local, procs);
	imprimir_vec("Vector leido: ", x_local, n_local, rank_local, procs);

	mat_vec_par(a_local, m, n, x_local, global_x, y_local, 
							m_local, n_local);
    	imprimir_vec("El producto es: ", y_local, m_local, rank_local, procs);

	MPI_Finalize();
	return 0;
}

void leer_matriz(char *entrada, MAT_LOCAL_T a_local, int m_local, int n,
						int rank_local, int procs) {
	int i;
	int j;
	int total_leidos;
	MAT_LOCAL_T temp;

	for (i = 0; i < procs*m_local; i++)
		for (j = n; j < MAX_N; j++)
			temp[i][j] = 0.0;

	if (rank_local == 0) {
		printf("%s\n", entrada);

		for (i = 0; i < procs*m_local; i++) 
			for (j = 0; j < n; j++)
				total_leidos = scanf("%f", &temp[i][j]);
	}

	MPI_Scatter(temp, m_local*MAX_N, MPI_FLOAT, a_local, m_local*MAX_N,
						MPI_FLOAT, 0, MPI_COMM_WORLD);
}

void leer_vector(char * entrada, float  *x_local, int n_local, int rank_local,
								int procs) {
	int i;
	int total_leidos;
	float temp[MAX_N];

	if (rank_local == 0) {
		printf("%s\n", entrada);

		for (i = 0; i < procs*n_local; i++) 
			total_leidos = scanf("%f", &temp[i]);
	}

	MPI_Scatter(temp, n_local, MPI_FLOAT, x_local, n_local, MPI_FLOAT,
							0, MPI_COMM_WORLD);
}

void mat_vec_par( MAT_LOCAL_T  a_local, int m, int n, float *x_local,
		float *global_x, float *y_local, int m_local, int n_local) {
	int i;
	int j;

	MPI_Allgather(x_local, n_local, MPI_FLOAT, global_x, n_local, MPI_FLOAT,
								MPI_COMM_WORLD);
	for (i = 0; i < m_local; i++) {
		y_local[i] = 0.0;
        
		for (j = 0; j < n; j++)
			y_local[i] = y_local[i] + a_local[i][j]*global_x[j];
	}
}

void imprimir_mat(char *intro, MAT_LOCAL_T a_local, int m_local, int n,
						int rank_local, int procs) {
	int i;
	int j;
	float temp[MAX_N][MAX_N];

	MPI_Gather(a_local, m_local*MAX_N, MPI_FLOAT, temp, m_local*MAX_N,
						MPI_FLOAT, 0, MPI_COMM_WORLD);

	if (rank_local == 0) {
		printf("%s\n", intro);
        
	for (i = 0; i < procs*m_local; i++) {
		for (j = 0; j < n; j++)
			printf("%4.1f ", temp[i][j]);
			printf("\n");
		}
	} 
}

void imprimir_vec(char *intro, float *y_local, int m_local, int rank_local,
								int procs) {
	int i;
	float temp[MAX_N];

	MPI_Gather(y_local, m_local, MPI_FLOAT, temp, m_local, MPI_FLOAT,
							0, MPI_COMM_WORLD);

	if (rank_local == 0) {
		printf("%s\n", intro);
	
		for (i = 0; i < procs*m_local; i++)
 			printf("%4.1f ", temp[i]);

		printf("\n");
	} 
}

