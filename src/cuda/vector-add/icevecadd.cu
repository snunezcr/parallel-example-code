/*
 * Instituto Tecnologico de Costa Rica
 * Centro de Investigaciones en Computacion
 * 
 * Asesoria-Practica en Computacion Paralela
 * Instituto Costarricense de Electricidad
 * Julio-Agosto 2011
 * 
 * Autor: Santiago Nunez Corrales
 * Programa: suma de vectores en CUDA C
 */

#include <stdio.h>
#define DEF_TAM 10

__global__ void suma_vec(float *a, float *b, float *c) {
	/* threadIdx.x es el identificador del hilo en la dimension x */
	int idx = threadIdx.x;
	a[idx] = 0;
	b[idx] = idx;
	c[idx] = a[idx] + b[idx];
}

int main (int argc, char *argv[]) {
	int n;
	int i;
	float a[DEF_TAM];
	float b[DEF_TAM];
	float c[DEF_TAM];
	float *ptr_dev_a;
	float *ptr_dev_b;
	float *ptr_dev_c;
	int mem_reservada;

	n = DEF_TAM;
	mem_reservada = DEF_TAM * sizeof(float);

	cudaMalloc((void **)&ptr_dev_a, mem_reservada);
	cudaMalloc((void **)&ptr_dev_b, mem_reservada);
	cudaMalloc((void **)&ptr_dev_c, mem_reservada);

	cudaMemcpy(ptr_dev_a, a, mem_reservada, cudaMemcpyHostToDevice);
	cudaMemcpy(ptr_dev_b, b, mem_reservada, cudaMemcpyHostToDevice);

	suma_vec<<<1, n>>>(ptr_dev_a, ptr_dev_b, ptr_dev_c);

	cudaMemcpy(c, ptr_dev_c, mem_reservada, cudaMemcpyDeviceToHost);

	for (i = 0; i < DEF_TAM; i++)
		printf("c[%d] = %f\n", i, c[i]);

	cudaFree(ptr_dev_a);
	cudaFree(ptr_dev_b);
	cudaFree(ptr_dev_c);

	return 0;
}


