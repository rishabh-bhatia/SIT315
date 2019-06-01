#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>
#include <CL/cl.h>

using namespace std;


#define MAX 4
#define NUM_THREADS 4
const int maxNum = 3;

cl_device_id device_id;
cl_context context;
cl_program program;
cl_kernel kernel;
cl_command_queue queue;

cl_event event = NULL;
cl_mem buferA, buferB, buferC;
int err;

// Create arrays
int arrA[MAX][MAX];
int arrB[MAX][MAX];
int arrC[MAX][MAX];

const int max = MAX;
const int TS = 4;
const size_t localId[2] = { TS, TS };
const size_t globalId[2] = { max, max };

cl_device_id create_device();
cl_program build_program(cl_context ctx, cl_device_id deviceId, const char* filename);
void setup_openCL_device_context_queue_kernel();
void setup_kernel_memory(int elements_per_process, int* rcvA, int* partC);
void copy_kernel_args();
void free_memory(); //Clear the memory used by OpenCL

int main()
{

	MPI_Init(NULL, NULL);

	//Calculating the number of chunks the matrice is getting divided into
	int np = 4;
	MPI_Comm_size(MPI_COMM_WORLD, &np);
	//Calculating numbr of elements per chunk
	int rows = MAX / np;
	int elementsInSlave = rows * MAX;

	int rank = 0;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	arrA[MAX][MAX] = arrB[MAX][MAX] = arrC[MAX][MAX] = { 0 };

	clock_t timerStart;
	//double start ;

	// Master will distribute to it's slaves
	if (rank == 0)
	{

		// Initialise Array A and B with random values
		for (int i = 0; i < MAX; i++)
		{
			for (int j = 0; j < MAX; j++)
			{
				arrA[i][j] = rand() % maxNum + 1;
			}
		}

		for (int i = 0; i < MAX; i++)
		{
			for (int j = 0; j < MAX; j++)
			{
				arrB[i][j] = rand() % maxNum + 1;
			}
		}

		// Print A and B
		cout << "A: " << endl;
		for (int i = 0; i < MAX; i++)
		{
			for (int j = 0; j < MAX; j++)
			{
				cout << arrA[i][j] << " ";
			}
			cout << endl;
		}
		cout << "B: " << endl;
		for (int i = 0; i < MAX; i++)
		{
			for (int j = 0; j < MAX; j++)
			{
				cout << arrB[i][j] << " ";
			}
			cout << endl;
		}

		timerStart = clock(); //Starting the timer
							  //auto start = chrono::steady_clock::now();
	}

	// Broadcast matrix B to all slave processes
	MPI_Bcast(&arrB, MAX * MAX, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Barrier(MPI_COMM_WORLD);

	//Prefill the parts of A and the array which will produce the final result i.e. arrC or resultC
	int* partOfA = new int[rows * MAX];
	int* resultC = new int[rows * MAX]{ 0 };

	// Scatter the A matrix
	MPI_Scatter(arrA, elementsInSlave, MPI_INT, partOfA, elementsInSlave, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Barrier(MPI_COMM_WORLD);

	//Multiplying the elements in each process and saving them in a matrice
	setup_openCL_device_context_queue_kernel();
	setup_kernel_memory(elementsInSlave, partOfA, resultC);
	copy_kernel_args();

	clEnqueueNDRangeKernel(queue, kernel, 2, NULL, globalId, localId, 0, NULL, &event);
	clWaitForEvents(1, &event);

	//Reading data from the device to arrC
	clEnqueueReadBuffer(queue, buferC, CL_TRUE, 0, elementsInSlave * sizeof(int), resultC, 0, NULL, NULL);
	free_memory();

	// MPI_GATHER the sol from each process
	MPI_Gather(resultC, elementsInSlave, MPI_INT, arrC, elementsInSlave, MPI_INT, 0, MPI_COMM_WORLD);
	//MPI_Barrier(MPI_COMM_WORLD);

	// Master processes the received parts (prints values)
	if (rank == 0)
	{
		//Stopping the timer
		clock_t end = clock();

		double time_taken = (double)(end)-(double)(timerStart) / CLOCKS_PER_SEC;

		/*auto end = chrono::steady_clock::now();
		auto time_taken = end - start;*/
		cout << "Time Taken :" << time_taken / 10 << " ms" << endl;

		cout << "C: " << endl;
		for (int i = 0; i < MAX; i++)
		{
			for (int j = 0; j < MAX; j++)
			{
				cout << arrC[i][j] << " ";
			}
			cout << endl;
		}
	}

	MPI_Finalize();

	return 0;
}

void free_memory()
{

	clReleaseKernel(kernel);
	clReleaseMemObject(buferA);
	clReleaseMemObject(buferB);
	clReleaseMemObject(buferC);

	clReleaseCommandQueue(queue);
	clReleaseProgram(program);
	clReleaseContext(context);
}

void copy_kernel_args()
{
	clSetKernelArg(kernel, 0, sizeof(int), (void*)& max);
	clSetKernelArg(kernel, 1, sizeof(int), (void*)& max);
	clSetKernelArg(kernel, 2, sizeof(int), (void*)& max);
	clSetKernelArg(kernel, 3, sizeof(cl_mem), (void*)& buferA);
	clSetKernelArg(kernel, 4, sizeof(cl_mem), (void*)& buferB);
	clSetKernelArg(kernel, 5, sizeof(cl_mem), (void*)& buferC);
	if (err < 0)
	{
		cout << "Error in creating kernel argument!" << endl;
		cout << "Error: " << err << endl;
		exit(1);
	}
}

void setup_kernel_memory(int elements_per_process, int* rcvA, int* partC)
{
	// Create memory buffers
	buferA = clCreateBuffer(context, CL_MEM_READ_ONLY, elements_per_process * sizeof(int), NULL, NULL);
	buferB = clCreateBuffer(context, CL_MEM_READ_ONLY, MAX * MAX * sizeof(int), NULL, NULL);
	buferC = clCreateBuffer(context, CL_MEM_READ_WRITE, elements_per_process * sizeof(int), NULL, NULL);

	// Copy matrices to the GPU
	clEnqueueWriteBuffer(queue, buferA, CL_TRUE, 0, elements_per_process * sizeof(int), rcvA, 0, NULL, NULL);
	clEnqueueWriteBuffer(queue, buferB, CL_TRUE, 0, MAX * MAX * sizeof(int), arrB, 0, NULL, NULL);
	clEnqueueWriteBuffer(queue, buferC, CL_TRUE, 0, elements_per_process * sizeof(int), partC, 0, NULL, NULL);
}

void setup_openCL_device_context_queue_kernel()
{
	device_id = create_device();
	cl_int err;
	context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &err);
	if (err < 0)
	{
		cout << "There was a problem in creating a Context!" << endl;
		exit(1);
	}

	program = build_program(context, device_id, "matrix_mul.cl");

	queue = clCreateCommandQueueWithProperties(context, device_id, 0, &err);
	if (err < 0)
	{
		cout << "Error in creating queue!" << endl;
		exit(1);
	};

	kernel = clCreateKernel(program, "multiply_matrices", &err);
	if (err < 0)
	{
		cout << "Can't create kernel!" << endl;
		cout << "Error: " << err;
		exit(1);
	};
}

cl_program build_program(cl_context ctx, cl_device_id deviceId, const char* filename)
{

	cl_program program;

	FILE* pFile;
	char* pBuf, * pLog;
	size_t pSize, lSize;
	errno_t err;

	// Read the matrix_mul.cl file and put it in pBuf
	err = fopen_s(&pFile, filename, "rb");
	if (pFile == NULL)
	{
		cout << "Can't find the setup file(Example: file.cl)!" << endl;
		exit(1);
	}
	fseek(pFile, 0, SEEK_END);
	pSize = ftell(pFile);
	rewind(pFile);
	pBuf = (char*)malloc(pSize + 1);
	pBuf[pSize] = '\0';
	fread(pBuf, sizeof(char), pSize, pFile);
	fclose(pFile);

	//Copies the file's array to pBuf and calls clCreateProgramWithSource
	program = clCreateProgramWithSource(ctx, 1, (const char**)& pBuf, &pSize, &err);

	if (err < 0)
	{
		cout << "Error in making a program!" << endl;
		exit(1);
	}
	free(pBuf);

	err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL); //Build program
	if (err < 0)
	{

		clGetProgramBuildInfo(program, deviceId, CL_PROGRAM_BUILD_LOG, 0, NULL, &lSize); // Find size of log file and print
		pLog = (char*)malloc(lSize + 1);
		pLog[lSize] = '\0';
		clGetProgramBuildInfo(program, deviceId, CL_PROGRAM_BUILD_LOG, lSize + 1, pLog, NULL);
		cout << pLog << endl;
		free(pLog);
		exit(1);
	}

	return program;
}

cl_device_id create_device()
{

	cl_platform_id platformId;
	cl_device_id deviceId;
	int err;

	err = clGetPlatformIDs(1, &platformId, NULL);//Get platform ID
	if (err < 0)
	{
		cout << "Error: Can't get Platform ID!" << endl;
		exit(1);
	}

	err = clGetDeviceIDs(platformId, CL_DEVICE_TYPE_GPU, 1, &deviceId, NULL);//Access device GPU
	if (err == CL_DEVICE_NOT_FOUND)
	{
		err = clGetDeviceIDs(platformId, CL_DEVICE_TYPE_CPU, 1, &deviceId, NULL);//Access device CPU
	}
	if (err < 0)
	{
		cout << "Error: Can't access CPU OR GPU!" << endl;
		exit(1);
	}

	return deviceId;
}