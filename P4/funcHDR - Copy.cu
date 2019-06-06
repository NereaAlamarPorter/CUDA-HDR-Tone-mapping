/* Code completed by Nerea Alamar & Antonio Marco Rodrigo*/

#include <cuda_runtime.h>
#include <iostream>
#include "device_launch_parameters.h"
#pragma once
#ifdef __INTELLISENSE__
void __syncthreads();
void atomicAdd(unsigned int*, unsigned int);
#endif

//Tamaño de bloque optimo para imagenes de resolución 2x1 (ej: waterfall_bg)
#define BLOCK_SIZE 128;

//Usamos checkCudaErrors como en la práctica anterior para detectar fallos de cuda
#define checkCudaErrors(val) check( (val), #val, __FILE__, __LINE__)

template<typename T>
void check(T err, const char* const func, const char* const file, const int line) {
	if (err != cudaSuccess) {
		std::cerr << "CUDA error at: " << file << ":" << line << std::endl;
		std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
		exit(1);
	}
}

//////////////////////////////////////////////////////////////////////////
//							   KERNELS									//
//////////////////////////////////////////////////////////////////////////


//////////////////////////////////////////////////////////////////////////
//							 SCAN KERNEL								//
//////////////////////////////////////////////////////////////////////////
/* Este kernel SCAN lleva a cabo un exclsuive scan mediante el metodo de Hillis Steele
de manera que movemos los valores a la derecha y ponemos un 0 en el principio para
que sea exclusive. Recibe como parametro de entrada el histograma generado en el paso
anterior, y consigue la distribucion acumulativa*/
__global__ void scan(unsigned int* output, const unsigned int* input, int numBins)
{
	//Obtenemos las id del thread con el que vamos a trabajar
	int i = threadIdx.x + blockDim.x*blockIdx.x;
	int thread = threadIdx.x;

	//usamos memoria compartida a través de la directiva __shared__
	extern __shared__ float shArrayS[];
	//creamos dos variables auxiliares las cuales valdrán 1 y 0 respectivamente, e irán
	//rotando su valor a lo largo del bucle for que realiza el scan
	int up = 0;
	int	down = 1;

	//exclusive scan (forzamos un 0 en el principio)
	if (thread > 0)
		shArrayS[thread] = input[thread - 1];
	else
		shArrayS[thread] = 0;

	// inclusive scan (ponemos el mismo valor que se encuentra en el histograma de entrada)
	// temp[tid] = d_in[tid];
	__syncthreads();

	// con este bucle realizamos el algoritmo de Hillis Steele, rellenando el array en memoria compartida
	for (int off = 1; off < numBins; off <<= 1) //usamos operaciones de bits para mejorar notablemente el rendimiento
	{
		up = 1 - up; // 1 <-> 0
		down = 1 - up; // 0 <-> 1
		if (thread >= off)
			shArrayS[numBins*up + thread] = shArrayS[numBins*down + thread] + shArrayS[numBins*down + thread - off];
		else
			shArrayS[numBins*up + thread] = shArrayS[numBins*down + thread];
		__syncthreads();
	}

	// una vez calculado, volcamos el array de memoria compartida en nuestro array de salida para devolver la distribucíon acumulativa
	output[thread] = shArrayS[up*numBins + thread];
}

//////////////////////////////////////////////////////////////////////////
//						   HISTOGRAMA KERNEL							//
//////////////////////////////////////////////////////////////////////////
/* Con este kernel calculamos el histograma dados: el numero de bins, los valores de rango de luminancias y sus valores minimo y maximo
(en concreto, solo nos hara falta el valor minimo y el rango)*/
__global__ void histo(unsigned int* output, const float * input, int numBins, int resolution, float lumMin, float lumRange)
{
	//Obtenemos las id del thread con el que vamos a trabajar
	int i = threadIdx.x + blockDim.x*blockIdx.x;
	int thread = threadIdx.x;

	//comprobamos si nos salimos de los limites de la imagen
	if (i >= resolution) 
		return;

	//Dividimos el valor del vector de luminancias por el nº de bin, dandote el bin en el que cae
	int bin = (input[i] - lumMin) / lumRange * numBins; 
	//suma atomica para que solo haga esta operacion un hilo a la vez (evita condiciones de carrera)
	atomicAdd(&(output[bin]), 1); 
}

//////////////////////////////////////////////////////////////////////////
//						   MAX & MIN FUNCTIONS							//
//////////////////////////////////////////////////////////////////////////
/* funciones que calculan el maximo y el minimo de dos valores dados, respectivamente.
Serán necesarias para el kernel de calculo de maximo y minimo valor de luminancia*/

//Devuelve el minimo de los dos valores pasados como parametro
__device__ float minimize(float a, float b)
{
	if (a < b)
		return a;
	else
		return b;
}

//Devuelve el maximo de los dos valores pasados como parametro
__device__ float maximize(float a, float b)
{
	if (a > b)
		return a;
	else
		return b;
}

//////////////////////////////////////////////////////////////////////////
//						    REDUCE KERNEL							    //
//////////////////////////////////////////////////////////////////////////
/*Este kernel calculara el valor maximo o minimo del vector de luminancias pasado como parametro.
Para evitar crear dos kernels practicamente iguales, le añadimos un tercer parametro bool llamado "greater",
cuyo valor decidira si lo que estamos calculando es el minimo o el maximo*/
__global__ void reduce(float* output, const float * input, bool greater) 
{
	//Obtenemos las id del thread con el que vamos a trabajar
	int i = threadIdx.x + blockDim.x*blockIdx.x;
	int thread = threadIdx.x;

	//usamos memoria compartida a través de la directiva __shared__
	extern __shared__ float shArrayR[];
	shArrayR[thread] = input[i];
	__syncthreads();

	//con este bucle realizamos un proceso de reduccion, en el cual el primer elemento
	//de nuestro vector en memoria compartida almacenara el valor maximo o minimo
	for (int s = blockDim.x >> 1; s > 0; s >>= 1) //usamos operaciones de bits para mejorar notablemente el rendimiento
	{
		if (thread < s) 
		{
			//si greater es true, calculamos el maximo, si no lo es, calculamos el minimo
			if (greater)
				shArrayR[thread] = maximize(shArrayR[thread], shArrayR[thread + s]);
			else
				shArrayR[thread] = minimize(shArrayR[thread], shArrayR[thread + s]);
		}
		__syncthreads();
	}

	//cuando estamos en el primer elemento, ya tendra almacenado el valor maximo o minimo, y lo devolvemos
	//en nuestro valor de salida
	if (thread == 0)
		output[blockIdx.x] = shArrayR[0];
}

//////////////////////////////////////////////////////////////////////////
//							   FUNCIONES								//
//////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////
//							 FUNCION MINMAX							    //
//////////////////////////////////////////////////////////////////////////
/*Esta funcion recibe como parametro el vector de luminancias, el tamaño de la imagen
y greater, que decidira si estamos buscando el minimo o el maximo. La funcion llamara al
kernel reduce para dicha tarea*/
float minmax(const float* const d_logLuminance, int resolution, bool greater) 
{
	int size = resolution;
	//definimos un vector half, que almacenara los valores intermedios del proceso de reduccion
	float* half = NULL;

	//numero de threads (tamaño de bloque)
	int blocksize = BLOCK_SIZE; 
	//numero de bloques (tamaño de grid)
	int gridsize = ceil(1.0f*size / blocksize);

	//cantida en bytes de memoria compartida que usara el kernel
	int sharedMemory = blocksize * sizeof(float);

	//en este bucle se realizara el proceso de reduccion al completo, a traves de sucesivas llamadas
	//al kernel reduce. Saldremos del mismo cuando solo tengamos relleno el elemento 1 del vector
	while (true) 
	{
		//definimos vector que contendra los resultados globales de la reduccion
		float* deviceResult;
		checkCudaErrors(cudaMalloc(&deviceResult, gridsize * sizeof(float)));

		//si estamos en la primera iteracion, pasamos al kernel el vector completo de luminancias
		if (half == NULL) 
			reduce << <gridsize, blocksize, sharedMemory >> > (deviceResult, d_logLuminance, greater);
		//si no, le pasamos el vector que contendra los resultados intermedios de la reduccion "half"
		else 
			reduce << <gridsize, blocksize, sharedMemory >> > (deviceResult, half, greater);
		cudaDeviceSynchronize();

		//vaciamos el vector intermedio y lo llenamos con el nuevo vector de valores intermedios obtenido
		if (half != NULL) 
			checkCudaErrors(cudaFree(half));
		half = deviceResult;

		//comprobamos si hemos llegado al final de la reduccion
		if (gridsize == 1) 
		{
			//copiamos el resultado de la reduccion a memoria principal (de CPU)
			float hostResult;
			checkCudaErrors(cudaMemcpy(&hostResult, deviceResult, sizeof(float), cudaMemcpyDeviceToHost));
			//salimos del bucle y devolvemos el valor maximo o minimo calculado
			return hostResult;
		}

		//actualizamos los valores disminuyendolos en cada iteracion, hasta llegar a 1
		size = gridsize;
		gridsize = ceil(1.0f*size / blocksize);
	}
}

//////////////////////////////////////////////////////////////////////////
//							 FUNCION HISTOGRAMA							//
//////////////////////////////////////////////////////////////////////////
/* Esta funcion recibe como parametros el vector de luminancias, el numero de bins, el tamaño, el valor minimo
de dicho vector calculado de la reduccion y el rango (calculado una vez obtenido el valor minimo y maximo) para
calcular el histograma mediante la llamada al kernel de histograma*/
unsigned int* histograma(const float* const d_logLuminance, int numBins, int resolution, float lumMin, float lumRange) 
{
	//numero de threads (tamaño de bloque)
	int blocksize = BLOCK_SIZE;
	//numero de bloques (tamaño de grid)
	int gridsize = ceil(1.0f*resolution / blocksize);

	//definimos el vector de enteros que contendra el histograma y devolveremos como salida
	unsigned int* result;
	//reservamos memoria e inicializamos los valores a 0
	checkCudaErrors(cudaMalloc(&result, numBins * sizeof(unsigned int)));
	checkCudaErrors(cudaMemset(result, 0, numBins * sizeof(unsigned int)));

	//llamamos al kernel para calcular el histograma
	histo << <gridsize, blocksize, 0 >> > (result, d_logLuminance, numBins, resolution, lumMin, lumRange);
	cudaDeviceSynchronize(); 

	return result;
}

//////////////////////////////////////////////////////////////////////////
//						FUNCION CALCULATE CDF						    //
//////////////////////////////////////////////////////////////////////////
/* Funcion principal que obtiene como parametros de entrada los valores de luminancia de la imagen,
el numero de filas y de columnas, y el numero de bins que usaremos para calcular el histograma. Esta
funcion se encargara de llamar a todas las funciones y kernels anteriores para obtener la distribucion
acumulativa siguiendo los pasos del TODO*/
void calculate_cdf(const float* const d_logLuminance,
	unsigned int* const d_cdf,
	float &min_logLum,
	float &max_logLum,
	const size_t numRows,
	const size_t numCols,
	const size_t numBins)
{
	/* TODO
	  1) Encontrar el valor máximo y mínimo de luminancia en min_logLum and max_logLum a partir del canal logLuminance
	  2) Obtener el rango a representar
	  3) Generar un histograma de todos los valores del canal logLuminance usando la formula
	  bin = (Lum [i] - lumMin) / lumRange * numBins
	  4) Realizar un exclusive scan en el histograma para obtener la distribución acumulada (cdf)
	  de los valores de luminancia. Se debe almacenar en el puntero c_cdf
	*/

	//1) Encontrar el valor máximo y mínimo de luminancia en min_logLum and max_logLum a partir del canal logLuminance

	//Definimos el tamaño de la imagen
	int resolution = numRows * numCols;
	//llamamos a la funcion minmax dos veces, pasandole un diferente valor para el parametro "greater", para calcular
	//el minimo y el maximo valor del vector de luminancias
	min_logLum = minmax(d_logLuminance, resolution, false);
	max_logLum = minmax(d_logLuminance, resolution, true);

	//2) Obtener el rango a representar

	//simplemente restamos el valor minimo del vector de luminancias al valor maximo del mismo
	float range = max_logLum - min_logLum;

	//3) Generar un histograma de todos los valores del canal logLuminance usando la formula
	//bin = (Lum[i] - lumMin) / lumRange * numBins

	//llamamos a la funcion histograma pasanole el valor minimo y el rango previamente calculados
	unsigned int* histoResult = histograma(d_logLuminance, numBins, resolution, min_logLum, range);

	//4) Realizar un exclusive scan en el histograma para obtener la distribución acumulada (cdf)
	//de los valores de luminancia.Se debe almacenar en el puntero c_cdf

	//llamamos directamente a nuestro scan kernel pasandole el histograma previamente calculado
	//obteniendo asi el valor final deseado
	scan << <1, numBins, 2 * numBins * sizeof(unsigned int) >> > (d_cdf, histoResult, numBins);

	checkCudaErrors(cudaFree(histoResult));
}