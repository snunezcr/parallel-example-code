/*
 * Instituto Tecnologico de Costa Rica
 * Centro de Investigaciones en Computacion
 * 
 * Asesoria-Practica en Computacion Paralela
 * Instituto Costarricense de Electricidad
 * Julio-Agosto 2011
 * 
 * Autor: Santiago Nunez Corrales
 * Programa: transferencia de calor 2D por ley de Fourier
 */

#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"

#define NXGRID 82
#define NYGRID 82
#define PASOS 300
#define MAX_P 10
#define MIN_P 3
#define INICIO 1
#define IZQTAG  2
#define DERTAG  3
#define VACIO  0
#define LISTO  4

void actualizar(int, int, int, float *, float *);
void init_datos(int, int, float *);
void guardar_datos(int, int, float *, char *);

struct Parms { 
  float cx;
  float cy;
} parms = {0.1, 0.1};

int main (int argc, char *argv[]) {
	float u[2][NXGRID][NYGRID];
	int rank_local;
	int trabajadores;
	int procs;
	int prom_filas;
	int filas;
	int desp;
	int extra;
	int destino;
	int fuente;
	int izquierda;
	int derecha;
	int tipo_mensaje;
	int inicio;
	int fin;
	int i;
	int ix;
	int iy;
	int iz;
	int it;
	MPI_Status estado;

	MPI_Init(&argc,&argv);
	MPI_Comm_size(MPI_COMM_WORLD,&procs);
	MPI_Comm_rank(MPI_COMM_WORLD,&rank_local);

	trabajadores = procs - 1;

	if (rank_local == 0) {
		if ((trabajadores > MAX_P) || (trabajadores < MIN_P)) {
			printf("Error. Deben existir entre  %d y %d tareas.\n",
                 					MIN_P + 1, MAX_P + 1);
         		printf("Abortando.\n");
         		MPI_Abort(MPI_COMM_WORLD, 0);
		 	exit(1);
		 }

		printf ("Iniciando Fourier 2D, %d procesos.\n", trabajadores);
		printf("Grid: X= %d  Y= %d  Pasos = %d\n",NXGRID,NYGRID,PASOS);
	      	printf("Inicializacion del grid y escritura de calor.dat\n");
		init_datos(NXGRID, NYGRID, (float *)u);
		guardar_datos(NXGRID, NYGRID, (float *)u, "calor.dat");

		prom_filas = NXGRID / trabajadores;
		extra = NXGRID % trabajadores;
		desp = 0;

		for (i=1; i<=trabajadores; i++) {
			filas = (i <= extra) ? prom_filas + 1 : prom_filas;

			if (i == 1)
				izquierda = VACIO;
			else
				izquierda = i - 1;

			if (i == trabajadores)
				derecha = VACIO;
			else
				derecha = i + 1;

			destino = i;

			MPI_Send(&desp, 1, MPI_INT, destino, INICIO, 
								MPI_COMM_WORLD);
		 	MPI_Send(&filas, 1, MPI_INT, destino, INICIO,
								MPI_COMM_WORLD);
			MPI_Send(&izquierda, 1, MPI_INT, destino, INICIO,
								MPI_COMM_WORLD);
			MPI_Send(&derecha, 1, MPI_INT, destino, INICIO,
								MPI_COMM_WORLD);
			MPI_Send(&u[0][desp][0], filas*NYGRID, MPI_FLOAT,
					destino, INICIO, MPI_COMM_WORLD);
		 
			printf("Enviado a tarea %d: filas= %d desp= %d ",
							destino, filas, desp);
			printf("izquierda= %d derecha= %d\n", 
							izquierda, derecha);

			desp = desp + filas;
	     	}

		for (i = 1; i <= trabajadores; i++) {
			fuente = i;
			tipo_mensaje = LISTO;

			MPI_Recv(&desp, 1, MPI_INT, fuente, tipo_mensaje,
						MPI_COMM_WORLD, &estado);
			MPI_Recv(&filas, 1, MPI_INT, fuente, tipo_mensaje,
						MPI_COMM_WORLD, &estado);
			MPI_Recv(&u[0][desp][0], filas*NYGRID, MPI_FLOAT,
				fuente, tipo_mensaje, MPI_COMM_WORLD, &estado);
		}

		printf("Guardando archivo final.dat\n");
		guardar_datos(NXGRID, NYGRID, &u[0][0][0], "final.dat");
	} else {
		for (iz = 0; iz < 2; iz++)
			for (ix = 0; ix < NXGRID; ix++) 
				for (iy = 0; iy < NYGRID; iy++) 
					u[iz][ix][iy] = 0.0;

		fuente = 0;
		tipo_mensaje = INICIO;

		MPI_Recv(&desp, 1, MPI_INT, fuente, tipo_mensaje,
						MPI_COMM_WORLD, &estado);
		MPI_Recv(&filas, 1, MPI_INT, fuente, tipo_mensaje,
						MPI_COMM_WORLD, &estado);
		MPI_Recv(&izquierda, 1, MPI_INT, fuente, tipo_mensaje,
						MPI_COMM_WORLD, &estado);
		MPI_Recv(&derecha, 1, MPI_INT, fuente, tipo_mensaje,
						MPI_COMM_WORLD, &estado);
		MPI_Recv(&u[0][desp][0], filas*NYGRID, MPI_FLOAT, fuente, 
					tipo_mensaje, MPI_COMM_WORLD, &estado);

		inicio = desp;
		fin = desp + filas-1;

		if (desp == 0) 
			inicio = 1;

		if ((desp+filas)==NXGRID)
			fin--;

		printf("proc=%d inicio=%d fin=%d\n", rank_local, inicio, fin);
		printf("Proc %d iniciando pasos con datos\n", rank_local);

		iz = 0;

		for (it = 1; it <= PASOS; it++) {
			if (izquierda != VACIO) {
				MPI_Send(&u[iz][desp][0], NYGRID, MPI_FLOAT, 
					izquierda, DERTAG, MPI_COMM_WORLD);

		    		fuente = izquierda;
				tipo_mensaje = IZQTAG;

				MPI_Recv(&u[iz][desp-1][0], NYGRID, MPI_FLOAT,
						fuente, tipo_mensaje,
						MPI_COMM_WORLD, &estado);
			}
			if (derecha != VACIO) {
		    		MPI_Send(&u[iz][desp+filas-1][0], NYGRID,
							MPI_FLOAT, derecha,
							IZQTAG, MPI_COMM_WORLD);

				fuente = derecha;
				tipo_mensaje = DERTAG;

				MPI_Recv(&u[iz][desp+filas][0], NYGRID,
					MPI_FLOAT, fuente, tipo_mensaje,
						MPI_COMM_WORLD, &estado);
			}

			actualizar(inicio, fin, NYGRID, &u[iz][0][0], 
								&u[1-iz][0][0]);
			iz = 1 - iz;
		}

		MPI_Send(&desp, 1, MPI_INT, 0, LISTO, MPI_COMM_WORLD);
		MPI_Send(&filas, 1, MPI_INT, 0, LISTO, MPI_COMM_WORLD);
		MPI_Send(&u[iz][desp][0], filas*NYGRID, MPI_FLOAT, 0, LISTO,
								MPI_COMM_WORLD);

	}
	MPI_Finalize();

	return 0;
}

void actualizar(int inicio, int fin, int ny, float *u1, float *u2) {
	int ix;
	int iy;

	for (ix = inicio; ix <= fin; ix++)
		for (iy = 1; iy <= ny - 2; iy++)
			*(u2+ix*ny+iy) = *(u1 + ix*ny + iy) + 
					parms.cx * (*(u1 + (ix + 1)*ny + iy) +
					*(u1 + (ix - 1)*ny + iy) - 
					2.0 * *(u1 + ix*ny + iy)) +
					parms.cy * (*(u1 + ix*ny + iy + 1) +
					*(u1 + ix*ny + iy - 1) -
					2.0 * *(u1+ix*ny+iy));
}

void init_datos(int nx, int ny, float *u) {
	int ix;
	int iy;

	for (ix = 0; ix <= nx-1; ix++)
		for (iy = 0; iy <= ny-1; iy++)
			*(u + ix*ny + iy) = (float)(ix * (nx - ix - 1) *
					iy * (ny - iy - 1));
}

void guardar_datos(int nx, int ny, float *u, char *arch) {
	int ix;
	int iy;
	FILE *fp;

	fp = fopen(arch, "w");

	for (iy = ny-1; iy >= 0; iy--)
		for (ix = 0; ix <= nx-1; ix++)
			fprintf(fp, "%d %d %6.4f\n", ix, iy, *(u + ix*ny + iy));

	fclose(fp);
}
