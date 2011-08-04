/*
 * Instituto Tecnologico de Costa Rica
 * Centro de Investigaciones en Computacion
 * 
 * Asesoria-Practica en Computacion Paralela
 * Instituto Costarricense de Electricidad
 * Julio-Agosto 2011
 * 
 * Autor: Santiago Nunez Corrales
 * Programa: ejemplo de profiling de una aplicacion en CUDA C
 */

#include <stdio.h>
#include <math.h>

#define MAX_DATOS 1024*1024*8
#define GRID_BLOQUE 128
#define HILOS_BLOQUE 256
#define SEMILLA 123

void prod_cpu(float *a, float *b, float *c, int count) {
	for (int i=0; i < count; i++) {
		c[count] = sqrt(a[count] * b[count] / 12.34567) * sin(a[count]);
	}
}

__global__ void prod_gpu(float *a, float *b, float *c) {
	int tid = (blockIdx.y * GRID_BLOQUE * HILOS_BLOQUE) + 
		blockIdx.x * HILOS_BLOQUE + threadIdx.x;

	c[tid] = sqrt(a[tid] * b[tid] / 12.34567) * sin(a[tid]);
}

int main(int argc, char *argv[]){
	float *host_a;
	float *host_b;
	float *host_c;
	float *dev_a;
	float *dev_b;
	float *dev_c;
	double t_gpu;
	int i;
	unsigned int temp_host;

	cutCreateTimer(&temp_host);
	printf("Inicializando datos...\n");
	
	host_a = (float *)malloc(sizeof(float) * MAX_DATOS);
	host_b = (float *)malloc(sizeof(float) * MAX_DATOS);
	host_c = (float *)malloc(sizeof(float) * MAX_DATOS);
	
	cudaMalloc((void **)&dev_a, sizeof(float) * MAX_DATOS);
	cudaMalloc( (void **)&dev_b, sizeof(float) * MAX_DATOS);
	cudaMalloc( (void **)&dev_c , sizeof(float) * MAX_DATOS);

	srand(SEMILLA);
	
	for(i = 0; i < MAX_DATOS; i++) {
		host_a[i] = (float)rand() / (float)RAND_MAX;
		host_b[i] = (float)rand() / (float)RAND_MAX;
	}

	int primera_vez = 1;
	const int usar_gpu = 1;

	for (int tot_datos = MAX_DATOS; tot_datos > 128*256; tot_datos /= 2) {
		int ancho_grid = 128;
		int alto_grid = (tot_datos / 256) / ancho_grid;

		dim3 blockGridRows(ancho_grid, alto_grid);
		dim3 threadBlockRows(256, 1);

		cutResetTimer(temp_host);
		cutStartTimer(temp_host);

		if (usar_gpu == 1) {

			cudaMemcpy(dev_a, host_a, sizeof(float) * tot_datos, 
							cudaMemcpyHostToDevice);
			cudaMemcpy(dev_b, host_b, sizeof(float) * tot_datos,
							cudaMemcpyHostToDevice);

			prod_gpu<<<blockGridRows, threadBlockRows>>>(dev_a, 
								dev_b, dev_c);

			cudaThreadSynchronize();
			cudaMemcpy(host_c, dev_c, sizeof(float) * tot_datos,
							cudaMemcpyDeviceToHost);
		} else {
			prod_cpu(host_a, host_b, host_c, tot_datos);
		}

		cutStopTimer(temp_host);
		t_gpu = cutGetTimerValue(temp_host);
		
		if (!primera_vez || !usar_gpu){
			printf("Elementos: %d -  tiempo: %f msec - " 
				"%f Multiplicaciones/seg\n", tot_datos, t_gpu,
				alto_grid * 128 * 256 / (t_gpu * 0.001));
		} else {
			primera_vez = 0;
			tot_datos *= 2;
		}
	}

	printf("Terminando...\n");
	cudaFree(dev_c);
	cudaFree(dev_b);
	cudaFree(dev_a);
	free(host_c);
	free(host_b);
	free(host_a);

	cutDeleteTimer(temp_host);
}

