/*
 * Instituto Tecnologico de Costa Rica
 * Centro de Investigaciones en Computacion
 * 
 * Asesoria-Practica en Computacion Paralela
 * Instituto Costarricense de Electricidad
 * Julio-Agosto 2011
 * 
 * Autor: Santiago Nunez Corrales
 * Programa: ejemplo basico de inicializacion de MPI
 */

#include <stdio.h>
#include <string.h>
#include "mpi.h"		/* Encabezado para accesar funciones de MPI */

int main(int argc, char *argv[]) {
	int rank_local;		/* Numero de proceso local */
	int procs;		/* Cantidad de procesos */
	int fuente;		/* Id de quien envia */
	int destino;		/* Id de quien recibe */
	int etiqueta;		/* Tipo de mensaje */
	char mensaje[200];	/* Cadena para guardar el mensaje */
	MPI_Status estado;	/* Manejador de estado para comunicaciones */

	/* Inicializar MPI con los argumentos de consola. Esta llamada se
	 * encarga internamente de replicar los datos, asi como de llevar
	 * argumentos especializados en caso de enviarse al compilador de MPI.
	 */
	MPI_Init(&argc, &argv);

	/* Identificar el numero de este proceso individual */
	MPI_Comm_rank(MPI_COMM_WORLD, &rank_local);

	/* Identificar la cantidad de procesos total para el programa */
	MPI_Comm_size(MPI_COMM_WORLD, &procs);

	/* En MPI, una forma estandar de trabajar los programas es inicializar
	 * valores en el proceso 0, llamado proceso maestro. Luego, este se
	 * encarga de distribuir la carga (idealmente de forma balanceada) 
	 * entre los demas procesos.
	 */
	if (rank_local != 0) {			/* Proceso esclavo */
		/* Crear el mensaje a enviar */
		sprintf(mensaje, "[PROC %d] Curso ICE!", rank_local);
		destino = 0;
		/* Considerar que toda cadena tiene un caracter nulo extra
		 * produce que la longitud del mensaje aumentada en 1.
		 */
		MPI_Send(mensaje, strlen(mensaje) + 1, MPI_CHAR, destino,
						etiqueta, MPI_COMM_WORLD);
	} else {				/* Proceso maestro */
		/* Recibir tantos mensajes como procesos esclavos haya
		 * e imprimirlos.
		 */
		for (fuente = 1; fuente < procs; fuente++) {
			MPI_Recv(mensaje, 200, MPI_CHAR, fuente, etiqueta,
						MPI_COMM_WORLD, &estado);
			printf("%s\n", mensaje);
		}
	}

	/* Todo programa que utilize MPI debe efectuar un cierre interno */
	MPI_Finalize();

	/* Retornar 0 en Unix/Linux indica que el programa termino sin 
	 * problemas.
	 */
	return 0;
}

