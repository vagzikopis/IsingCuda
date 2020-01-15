#define BLOCK_SIZE 32
#define GRID_SIZE 32
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <math.h>
#include "gputimer.h"

__global__ void cudaKernel(int n, double* gpuWeights, int* gpuG, int* gpuTempGrid, int *flag)
{
	// Moment's coordinates in the grid //
  // allocate shared memory for weights
  int momentCol = blockIdx.x*blockDim.x + threadIdx.x;
	int	momentRow = blockIdx.y*blockDim.y + threadIdx.y;

  int gridRowIdx, gridColIdx;
  // Variable storing the total neighbourhood influence //
  double weightFactor = 0.0;
	// Each thread calculates the spin for a block of moments //
  // The step is based on the GRID_SIZE and BLOCK_SIZE //
  for(int i=momentRow; i<n; i+=blockDim.y*gridDim.y)
	{
		for(int j=momentCol; j<n; j+=blockDim.x*gridDim.x)
		{
		  weightFactor = 0.0;
      // Read 24 neighbours of every moment and calculate their total influence //
			for(int weightsRow=0; weightsRow<5; weightsRow++)
			{
				for(int weightsCol=0; weightsCol<5; weightsCol++)
				{
					if(weightsCol==2 && weightsRow==2)
					continue;
          // Calculate neighbour's coordinates in G //
          // using modulus to satisfy boundary conditions //
					gridRowIdx = (weightsRow - 2 + i + n) % n;
					gridColIdx = (weightsCol - 2 + j + n) % n;

					weightFactor+= gpuG[gridRowIdx * n + gridColIdx] * gpuWeights[weightsRow*5+weightsCol];
				}
			}
      // Update moment's atomic spin //
      // Set flag if a spin value transition has been done //
			if(weightFactor < 0.0001 && weightFactor > -0.0001)
			{
				gpuTempGrid[n*i+j] = gpuG[n*i+j];
			}else if(weightFactor > 0.00001)
			{
				gpuTempGrid[n*i+j] = 1;
        *flag = 1;
			}else
			{
				gpuTempGrid[n*i+j] = -1;
        *flag = 1;
			}
		}
	}
}

void ising( int *G, double *w, int k, int n)
{
  GpuTimer timer;
  FILE *fptr;
  fptr = fopen("v2data.csv","a");
  if(fptr==NULL)
    perror("File Error");
	// Array to store weights in GPU memory //
  double *gpuWeights;
	// Array to store G in GPU memory //
	int *gpuG;
	// Array used for ising calculations in GPU //
	int *gpuTempGrid;
	// Variable used for pointers swapping //
	int *gpuTempPtr;
  // Variables used to stop updating if the lattice remains the same //
  int flag;
  int *gpuFlag;
  // Start timing using Cuda Events //
  // Functions are written in gputimer.h //
  timer.Start();
  // cudaMalloc and cudaMemcpy used to allocate memory space //
  // and transfer data to the GPU //
  cudaMalloc(&gpuFlag, sizeof(int));
  cudaMalloc(&gpuWeights, 25*sizeof(double));
  cudaMemcpy(gpuWeights, w, 25*sizeof(double), cudaMemcpyHostToDevice);
  cudaMalloc(&gpuG, n*n*sizeof(int));
  cudaMemcpy(gpuG, G, n*n*sizeof(int), cudaMemcpyHostToDevice);
  cudaMalloc(&gpuTempGrid, n*n*sizeof(int));
  // Set Grid and Block dimensions
	dim3 dimGrid(GRID_SIZE,GRID_SIZE);
	dim3 dimBlock(BLOCK_SIZE,BLOCK_SIZE);
  for(int i = 0; i < k; i++){
    flag=0;
		cudaMemcpy(gpuFlag, &flag, sizeof(int), cudaMemcpyHostToDevice);
		// Kernel function is executed by every thread //
		cudaKernel<<<dimGrid , dimBlock>>>(n, gpuWeights, gpuG, gpuTempGrid, gpuFlag);

		// Thread synchronization. Wait for every thread to be executed //
		// befora proceeding to the next iteration //
		cudaDeviceSynchronize();

		// Swap gpu_G and gpu_gTemp pointers for next iteration to avoid copying data on every iteration //
		gpuTempPtr = gpuG;
		gpuG = gpuTempGrid;
		gpuTempGrid = gpuTempPtr;

    cudaMemcpy(&flag, gpuFlag, sizeof(int), cudaMemcpyDeviceToHost);
    // Break if the lattice remained the same as before //
    // if(flag==0)
    //   break;
	}
	// Transfer data back to CPU memory and store them in G //
  cudaMemcpy(G, gpuG, n*n*sizeof(int), cudaMemcpyDeviceToHost);
  // Stop timing //
	timer.Stop();
	fprintf(fptr,"%g,%d,%d,%d,%d\n", timer.Elapsed(),n ,k,GRID_SIZE,BLOCK_SIZE);
  fclose(fptr);
	// Free allocated memory fro the GPU //
  cudaFree(gpuG);
	cudaFree(gpuTempGrid);
	cudaFree(gpuWeights);

}
