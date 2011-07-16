/*
 * Instituto Tecnologico de Costa Rica
 * Centro de Investigaciones en Computacion
 * 
 * Asesoria-Practica en Computacion Paralela
 * Instituto Costarricense de Electricidad
 * Julio-Agosto 2011
 * 
 * Autor: Santiago Nunez Corrales
 * Programa: integracion por regla trapezoidal (Riemann)
 */

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "mpi.h"

#define DEF_AB 	1.0	/* Distancia por default para a - b */
#define DEF_N	100	/* Puntos de integracion por default */

/* Prototipos de funciones */
float trapezoide(float, float, int, float);
float f(float);			/* Esta es la funcion que se integrara */

int main(int argc, char *argv[]) {
	int rank_local;
	int procs;
	float a;
	float b;
	int n;
	float h;
	float mi_a;
	float mi_b;
	int mi_n;
	float mi_trapezoide;
	float total;
	int fuente;
	int destino = 0;
	int etiqueta = 0;
	MPI_Status estado;

	/* Se obtienen los valores iniciales desde consola y se validan. En caso
	 * de que la validacion falle, se utilizan defaults
	 */
	if (argc != 4) {
		fprintf(stderr, "Error: cantidad incorrecta de parametros");
		return -1;
	}

	a = atof(argv[1]);
	b = atof(argv[2]);

	/* Si se integra sobre un intervalo de longitud 0, coloque valores por
	 * default.
	 */
	if (fabs(a - b) == 0) {
		a = 0;
		b = DEF_AB;
	}

	n = atoi(argv[3]);
	
	/* Si la cantidad de pasos es menor a 1, asigne por default 100 */
	if (n < 1)
		n = DEF_N;

	/* Inicializar MPI */
	MPI_Init(&argc, &argv);

	/* Obtener id de este proceso */
	MPI_Comm_rank(MPI_COMM_WORLD, &rank_local);

	/* Obtener el numero total de procesos */
	MPI_Comm_size(MPI_COMM_WORLD, &procs);

	/* Se calculan los datos que todos los pasos de integracion necesitan */
	h = (b - a)/n;
	mi_n = n/procs;

	/* La longitud del intervalo de integracion para cada proceso esta
	 * dada en funcion de mi_n y h*/
	mi_a = a + rank_local*mi_n*h;
	mi_b = mi_a + mi_n*h;
	mi_trapezoide = trapezoide(mi_a, mi_b, mi_n, h);

	/* Se calcula el resultado de la suma de Riemann con cada una de las
	 * reglas trapezoidales.
	 */
	if (rank_local == 0) {
		total = mi_trapezoide;
		for (fuente = 1; fuente < procs; fuente++) {
			MPI_Recv(&mi_trapezoide, 1, MPI_FLOAT, fuente, etiqueta,
						MPI_COMM_WORLD, &estado);
			total += mi_trapezoide;
		}
	} else {
		MPI_Send(&mi_trapezoide, 1, MPI_FLOAT, destino, etiqueta,
								MPI_COMM_WORLD);
	}

	/* Se imprime el resultado */
	if (rank_local == 0)
		printf("Con %d trapezoides se estima la intregral por sumas de"
			" Riemann\n desde %f hasta %f = %f\n", n, a ,b, total);

	/* Se finaliza el programa MPI */
	MPI_Finalize();
	
	return 0;
}

/* Funcion que calcula la regla trapezoidal */
float trapezoide(float mi_a, float mi_b, int mi_n, float h) {
	float resultado;
	float x;
	int i;

	resultado = (f(mi_a) + f(mi_b))/2;
	x = mi_a;

	for (i = 1; i < mi_n; i++) {
		x = x + h;
		resultado = resultado + f(x);
	}

	resultado = resultado*h;	

	return resultado;
}

/* Funcion a integrar. En este caso se integra x^2 - log(x) */
float f(float x) {
	float resultado;

	resultado = x*x - 2*x;

	return resultado;
}

