/*
 * Instituto Tecnologico de Costa Rica
 * Centro de Investigaciones en Computacion
 * 
 * Asesoria-Practica en Computacion Paralela
 * Instituto Costarricense de Electricidad
 * Julio-Agosto 2011
 * 
 * Autor: Santiago Nunez Corrales
 * Programa: sort bitonico en CUDA C
 */


#include <math.h>

#define DEF_TAM 64

__global__ void comparar( int* b, int* a, int largo, int offset, int N) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int mi_valor;

	a[idx] = b[idx];

	if (((idx/largo)%2) == 0)
		if (((idx/offset)%2) == 0)
			if (a[idx] < a[idx + offset])
				mi_valor = a[idx];
			else
				mi_valor = a[idx + offset];
		else
			if (a[idx] < a[idx - offset])
				mi_valor = a[idx - offset];
			else
				mi_valor = a[idx];
	else
		if (((idx/offset)%2) == 0)
			if (a[idx] < a[idx + offset])
				mi_valor = a[idx + offset];
			else
				mi_valor = a[idx];
		else
			if (a[idx] < a[idx - offset])
				mi_valor = a[idx];
			else
				mi_valor = a[idx - offset];

	b[idx] = mi_valor;
}

int main(int argc, char *argv[]) {
	int *a;
	int *b;
	int *c;
	int i;
	int largo;
	int offset

	cudaMalloc((void**)&a, sizeof(int)*DEF_TAM);
	cudaMalloc((void**)&b, sizeof(int)*DEF_TAM);

	c = (int*) malloc(sizeof(int) * 64);

	for (i = 0; i < DEF_TAM; i++)
		c[i] = random() % 1024;

	for (i = 0; i < DEF_TAM; i++)
		printf(" %d", c[i]);
	printf("\n");

	cudaMemcpy(b, c, sizeof(int)*DEF_TAM, cudaMemcpyHostToDevice);

	for (largo = 2; largo <= DEF_TAM; largo+= largo)
		for (offset = largo/2; offset > 0; offset /= 2)
			comparar<<<1, DEF_TAM>>>(b, a, largo, offset, DEF_TAM);


	cudaMemcpy(c, b, sizeof(int)*DEF_TAM, cudaMemcpyDeviceToHost));

	for (int i = 0; i < DEF_TAM; ++i)
		printf(" %d ", c[i]);
	printf("\n");

	cudaFree(a);
	cudaFree(b);
	free(c);

	return 0;
}

