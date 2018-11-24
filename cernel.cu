
#include "C:\Users\Alex\source\repos\librariesFromInternet\headers\stdafx.h"
#include <GL/freeglut.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <complex>

//cuda++
#include <cuda_runtime.h>
#include <cufft.h>
#include <cufftXt.h>
#include <helper_cuda.h>
#include <helper_functions.h>
//cuda--

#define signalCount 3
#define DIM 256
static float signal[DIM];
static float convolvedSignal[DIM];
float spectrumX[DIM];
float spectrumY[DIM];
static float phase[DIM];

//#define NX 256
#define BATCH 1
#define RANK 1

//cuda++
// Complex data type
typedef float2 Complex;
static __device__ __host__ inline Complex ComplexAdd(Complex, Complex);
static __device__ __host__ inline Complex ComplexScale(Complex, float);
static __device__ __host__ inline Complex ComplexMul(Complex, Complex);
static __global__ void ComplexPointwiseMulAndScale(Complex *, const Complex *,
	int, float);

// Filtering functions
void Convolve(const Complex *, int, const Complex *, int, Complex *);

// Padding functions
int PadData(const Complex *, Complex **, int, const Complex *, Complex **, int);

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
//void runTest2(int argc, char **argv);

// The filter size is assumed to be a number smaller than the signal size
#define SIGNAL_SIZE 256//50
#define FILTER_KERNEL_SIZE 52//11
//cuda--

void signalGeneration()
{
	float a[signalCount] = { 1, 3.5, 1.2 };
	float b[signalCount] = { 0, 2.9, 3.2 };
	float k[signalCount] = { 25, 50, 75 };
	int i = 0;
	int x;
	float fx;

	for (x = 0; x < DIM; x++)
	{
		signal[x] = 0;
		convolvedSignal[x] = 0;
		spectrumX[x] = 0;
		spectrumY[x] = 0;
	}
	for (i = 0; i < signalCount; i++)
	{
		for (x = 0; x < DIM; x++)
		{
			fx = x;
			signal[x] += (a[i] * sin(k[i] * fx / DIM) + b[i] * cos(k[i] * fx / DIM)) / (a[i] + b[i]) / signalCount;
		}
	}
}

void glinit(void)
{
	GLfloat values[2];
	glGetFloatv(GL_LINE_WIDTH_GRANULARITY, values);
	printf("GL_LINE_WIDTH_GRANULARITY value is %3.1f\n", values[0]);

	glGetFloatv(GL_LINE_WIDTH_RANGE, values);
	printf("GL_LINE_WIDTH_RANGE values are %3.1f %3.1f\n",
		values[0], values[1]);

	glEnable(GL_LINE_SMOOTH);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glHint(GL_LINE_SMOOTH_HINT, GL_DONT_CARE);
	glLineWidth(1.5);

	glClearColor(0.0, 0.0, 0.0, 0.0);
}

void display(void)
{
	int x;
	float fx;
	glClear(GL_COLOR_BUFFER_BIT);
	//glColor3f(0.0, 1.0, 0.0);
	glPushMatrix();

	glBegin(GL_POINTS);

	for (x=0;x<DIM;x++)
	{
		fx = x;
		glColor3f(0.0, 1.0, 0.0);
		//glVertex2f(2 * fx / DIM - 1, signal[x]);//original Green
		//glColor3f(1.0, 0.0, 0.0);
		//glVertex2f(2 * fx / DIM - 1, convolvedSignal[x] / 1.0e+36);//calculated Red
		//glColor3f(0.0, 0.5, 0.5);
		if (x<33) glVertex2f(spectrumX[x] / 12+ spectrumY[x] / 12,2*fx/DIM-1);//spectrum X
		//glColor3f(0.5, 0.5, 0.0);
		//glColor3f(1.0, 0.0, 0.0);
		//if (x<33) glVertex2f(2 * fx / DIM - 1, spectrumY[x] / 12);//spectrum Y
		//glVertex2f(2 * fx / DIM - 1, spectrumY[x] / 1.0e+37);//spectrum Y
	}
	glEnd();

	glPopMatrix();
	glFlush();
}



void drawing(int argc, char** argv)
{
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB);
	glutInitWindowSize(256, 256);
	glutCreateWindow(argv[0]);
	glinit();
	glutDisplayFunc(display);
	glutMainLoop();
}

void runTest2()
{
	printf("[cufft] is starting...\n");
	cufftHandle		plan;
	cufftComplex	*deviceDataComplex;
	cufftReal		*deviceDataReal;
	//cufftComplex	o_data[SIGNAL_SIZE*BATCH];
	Complex *o_data =
		reinterpret_cast<Complex *>(malloc(sizeof(Complex) * SIGNAL_SIZE*BATCH));

	//// Allocate host memory for the signal
	//cufftReal *hi_data =
	//	reinterpret_cast<Complex *>(malloc(sizeof(Complex) * SIGNAL_SIZE));



	checkCudaErrors(cudaMalloc((void**)&deviceDataComplex, sizeof(cufftComplex)*(SIGNAL_SIZE / 2 + 1)*BATCH));
	checkCudaErrors(cudaMalloc((void**)&deviceDataReal, sizeof(cufftReal)*SIGNAL_SIZE*BATCH));
	//checkCudaErrors(cudaMalloc((void**)&o_data, sizeof(cufftComplex)*(SIGNAL_SIZE / 2 + 1)*BATCH));
	//// Initialize the memory for the signal
	//for (unsigned int i = 0; i < SIGNAL_SIZE; ++i) {
	//	hi_data[i].x = signal[i];
	//	hi_data[i].y = 0;
	//}

	checkCudaErrors(cudaMemcpy(deviceDataReal, signal, SIGNAL_SIZE*BATCH, cudaMemcpyHostToDevice));

	if (cufftPlan1d(&plan, SIGNAL_SIZE, CUFFT_R2C, BATCH) != CUFFT_SUCCESS) {
		fprintf(stderr, "CUFFT error: Plan creation failed");
		//return void;
	}

	/* Use the CUFFT plan to transform the signal in place. */
	if (cufftExecR2C(plan, deviceDataReal, deviceDataComplex) != CUFFT_SUCCESS) {
		fprintf(stderr, "CUFFT error: ExecC2C Forward failed");
		//return 2;
	}

	if (cudaDeviceSynchronize() != cudaSuccess) {
		fprintf(stderr, "Cuda error: Failed to synchronize\n");
		//return 3;
	}

	checkCudaErrors(cudaMemcpy(o_data, deviceDataComplex, SIGNAL_SIZE*BATCH, cudaMemcpyDeviceToHost));

	//lex++
	
	//for (int i = 0; i < SIGNAL_SIZE; i++)
	//{
	//	lexResult[i].x = 0;
	//	lexResult[i].y = 0;
	//}
	//checkCudaErrors(cudaMemcpy(lexResult, d_signal, sizeof(Complex) * SIGNAL_SIZE,
	//	cudaMemcpyDeviceToHost));
	
	for (int i = 0; i < SIGNAL_SIZE; i++)
	{
		spectrumX[i] = o_data[i].x;
		spectrumY[i] = o_data[i].y;
	}
	//lex--


	//for (int i = 0; i < 32; i++)
	//{
	//	spectrumX = 0;// o_data[i].x;
	//	spectrumY = 0;// o_data[i].y;
	//}

	checkCudaErrors(cufftDestroy(plan));
	checkCudaErrors(cudaFree(deviceDataReal));
	checkCudaErrors(cudaFree(deviceDataComplex));	
	//checkCudaErrors(cudaFree(o_data));
	//return 0;
}

int main(int argc, char** argv)
{
	signalGeneration();

	//cuda++
	runTest2();
	//cuda--

	drawing(argc, argv);

	return 0;
}



//cuda++
////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
//void runTest(int argc, char **argv) {
//	printf("[simpleCUFFT] is starting...\n");
//
//	findCudaDevice(argc, (const char **)argv);
//
//	// Allocate host memory for the signal
//	Complex *h_signal =
//		reinterpret_cast<Complex *>(malloc(sizeof(Complex) * SIGNAL_SIZE));
//
//	// Initialize the memory for the signal
//	for (unsigned int i = 0; i < SIGNAL_SIZE; ++i) {
//		h_signal[i].x = signal[i];//sin(fi / 10) * 50;
//		h_signal[i].y = 0;
//	}
//
//	//// Allocate host memory for the filter
//	//Complex *h_filter_kernel =
//	//	reinterpret_cast<Complex *>(malloc(sizeof(Complex) * FILTER_KERNEL_SIZE));
//
//	//// Initialize the memory for the filter
//	//for (unsigned int i = 0; i < FILTER_KERNEL_SIZE; ++i) {
//	//	//h_filter_kernel[i].x = rand() / static_cast<float>(RAND_MAX);
//	//	//h_filter_kernel[i].x = sin(i / 10) * 50;
//	//	h_filter_kernel[i].x = 1;
//	//	h_filter_kernel[i].y = 0;
//	//	//printf("fil %d: %f\n", i, h_filter_kernel[i].x);
//	//}
//
//	// Pad signal and filter kernel
//	Complex *h_padded_signal;
//	Complex *h_padded_filter_kernel;
//	int new_size =
//		PadData(h_signal, &h_padded_signal, SIGNAL_SIZE, h_filter_kernel,
//			&h_padded_filter_kernel, FILTER_KERNEL_SIZE);
//	int mem_size = sizeof(Complex) * new_size;
//
//	// Allocate device memory for signal
//	Complex *d_signal;
//	//checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_signal), mem_size));
//	checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_signal), mem_size));
//	// Copy host memory to device
//	checkCudaErrors(
//		cudaMemcpy(d_signal, h_padded_signal, mem_size, cudaMemcpyHostToDevice));
//	checkCudaErrors(
//		cudaMemcpy(d_signal, h_signal, mem_size, cudaMemcpyHostToDevice));
//
//	// Allocate device memory for filter kernel
//	Complex *d_filter_kernel;
//	checkCudaErrors(
//		cudaMalloc(reinterpret_cast<void **>(&d_filter_kernel), mem_size));
//
//	// Copy host memory to device
//	checkCudaErrors(cudaMemcpy(d_filter_kernel, h_padded_filter_kernel, mem_size,
//		cudaMemcpyHostToDevice));
//
//	// CUFFT plan simple API
//	cufftHandle plan;
//	//checkCudaErrors(cufftPlan1d(&plan, new_size, CUFFT_C2C, 1));
//	checkCudaErrors(cufftPlan2d(&plan, NX, NY, CUFFT_C2C));
//
//	checkCudaErrors(cufftExecC2C(plan, reinterpret_cast<cufftComplex *>(d_signal),
//		reinterpret_cast<cufftComplex *>(d_signal),
//		CUFFT_FORWARD));
//	// Transform signal and kernel--
//
//	//lex++
//	Complex *lexResult =
//		reinterpret_cast<Complex *>(malloc(sizeof(Complex) * SIGNAL_SIZE));
//	for (int i = 0; i < SIGNAL_SIZE; i++)
//	{
//		lexResult[i].x = 0;
//		lexResult[i].y = 0;
//	}
//	checkCudaErrors(cudaMemcpy(lexResult, d_signal, sizeof(Complex) * SIGNAL_SIZE,
//		cudaMemcpyDeviceToHost));
//
//	for (int i = 0; i < SIGNAL_SIZE; i++)
//	{
//		spectrumX[i] = lexResult[i].x;
//		spectrumY[i] = lexResult[i].y;
//	}
//	//lex--
//
//	// Multiply the coefficients together and normalize the result
//	printf("Launching ComplexPointwiseMulAndScale<<< >>>\n");
//	ComplexPointwiseMulAndScale << <32, 256 >> >(d_signal, d_filter_kernel, new_size,
//		1.0f / new_size);
//
//	// Check if kernel execution generated and error
//	getLastCudaError("Kernel execution failed [ ComplexPointwiseMulAndScale ]");
//
//	// Transform signal back
//	printf("Transforming signal back cufftExecC2C\n");
//	checkCudaErrors(cufftExecC2C(plan, reinterpret_cast<cufftComplex *>(d_signal),
//		reinterpret_cast<cufftComplex *>(d_signal),
//		CUFFT_INVERSE));
//
//	// Copy device memory to host
//	Complex *h_convolved_signal = h_padded_signal;
//	checkCudaErrors(cudaMemcpy(h_convolved_signal, d_signal, mem_size,
//		cudaMemcpyDeviceToHost));
//
//	for (unsigned int i = 0; i < new_size; ++i) {
//		//printf("i:%d c:%f\n", i, h_convolved_signal[i].x);
//		convolvedSignal[i] = h_convolved_signal[i].x;
//	}
//
//	// Allocate host memory for the convolution result
//	Complex *h_convolved_signal_ref =
//		reinterpret_cast<Complex *>(malloc(sizeof(Complex) * SIGNAL_SIZE));
//
//	// Convolve on the host
//	Convolve(h_signal, SIGNAL_SIZE, h_filter_kernel, FILTER_KERNEL_SIZE,
//		h_convolved_signal_ref);
//
//	// check result
//	bool bTestResult = sdkCompareL2fe(
//		reinterpret_cast<float *>(h_convolved_signal_ref),
//		reinterpret_cast<float *>(h_convolved_signal), 2 * SIGNAL_SIZE, 1e-5f);
//
//	// Destroy CUFFT context
//	checkCudaErrors(cufftDestroy(plan));
//	//checkCudaErrors(cufftDestroy(plan_adv));
//
//	// cleanup memory
//	free(lexResult);
//	free(h_signal);
//	free(h_filter_kernel);
//	free(h_padded_signal);
//	free(h_padded_filter_kernel);
//	free(h_convolved_signal_ref);
//	checkCudaErrors(cudaFree(d_signal));
//	checkCudaErrors(cudaFree(d_filter_kernel));
//	//char str[80];
//	//scanf("%79s", str);
//	//exit(bTestResult ? EXIT_SUCCESS : EXIT_FAILURE);
//}

// Pad data
int PadData(const Complex *signal, Complex **padded_signal, int signal_size,
	const Complex *filter_kernel, Complex **padded_filter_kernel,
	int filter_kernel_size) {
	int minRadius = filter_kernel_size / 2;
	int maxRadius = filter_kernel_size - minRadius;
	int new_size = signal_size + maxRadius;

	// Pad signal
	Complex *new_data =
		reinterpret_cast<Complex *>(malloc(sizeof(Complex) * new_size));
	memcpy(new_data + 0, signal, signal_size * sizeof(Complex));
	memset(new_data + signal_size, 0, (new_size - signal_size) * sizeof(Complex));
	*padded_signal = new_data;

	// Pad filter
	new_data = reinterpret_cast<Complex *>(malloc(sizeof(Complex) * new_size));
	memcpy(new_data + 0, filter_kernel + minRadius, maxRadius * sizeof(Complex));
	memset(new_data + maxRadius, 0,
		(new_size - filter_kernel_size) * sizeof(Complex));
	memcpy(new_data + new_size - minRadius, filter_kernel,
		minRadius * sizeof(Complex));
	*padded_filter_kernel = new_data;

	return new_size;
}

////////////////////////////////////////////////////////////////////////////////
// Filtering operations
////////////////////////////////////////////////////////////////////////////////

// Computes convolution on the host
void Convolve(const Complex *signal, int signal_size,
	const Complex *filter_kernel, int filter_kernel_size,
	Complex *filtered_signal) {
	int minRadius = filter_kernel_size / 2;
	int maxRadius = filter_kernel_size - minRadius;

	// Loop over output element indices
	for (int i = 0; i < signal_size; ++i) {
		filtered_signal[i].x = filtered_signal[i].y = 0;

		// Loop over convolution indices
		for (int j = -maxRadius + 1; j <= minRadius; ++j) {
			int k = i + j;

			if (k >= 0 && k < signal_size) {
				filtered_signal[i] =
					ComplexAdd(filtered_signal[i],
						ComplexMul(signal[k], filter_kernel[minRadius - j]));
			}
		}
	}
}

////////////////////////////////////////////////////////////////////////////////
// Complex operations
////////////////////////////////////////////////////////////////////////////////

// Complex addition
static __device__ __host__ inline Complex ComplexAdd(Complex a, Complex b) {
	//char str[80];
	Complex c;
	c.x = a.x + b.x;
	c.y = a.y + b.y;
	//scanf("%79s", str);
	return c;
}

// Complex scale
static __device__ __host__ inline Complex ComplexScale(Complex a, float s) {
	//char str[80];
	Complex c;
	c.x = s * a.x;
	c.y = s * a.y;
	//scanf("%79s", str);
	return c;
}

// Complex multiplication
static __device__ __host__ inline Complex ComplexMul(Complex a, Complex b) {
	//char str[80];
	Complex c;
	c.x = a.x * b.x - a.y * b.y;
	c.y = a.x * b.y + a.y * b.x;
	//scanf("%79s", str);
	return c;
}

// Complex pointwise multiplication
static __global__ void ComplexPointwiseMulAndScale(Complex *a, const Complex *b,
	int size, float scale) {
	const int numThreads = blockDim.x * gridDim.x;
	const int threadID = blockIdx.x * blockDim.x + threadIdx.x;

	for (int i = threadID; i < size; i += numThreads) {
		a[i] = ComplexScale(ComplexMul(a[i], b[i]), scale);
	}
}
//cuda--
