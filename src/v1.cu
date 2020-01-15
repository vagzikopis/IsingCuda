#define BLOCK_SIZE 8
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../inc/gputimer.h"

__global__ void cudaKernel(int n, double* gpuWeights, int* gpuG, int* gpuTempGrid, int* flag)
{
	// Moment's coordinates in the grid //
	int momentRow = blockIdx.y*blockDim.y + threadIdx.y;
	int momentCol = blockIdx.x*blockDim.x + threadIdx.x;
  int gridRowIdx, gridColIdx;

	// Variable storing the total neighbourhood influence //
	double weightFactor = 0.0;

	// Check if coordinates are valid //
	if(momentRow < n && momentCol < n){
		// Read 24 neighbours of every moment and calculate their total influence //
    for(int row=0; row<5; row++)
    {
      for(int col=0; col<5; col++)
      {
        if(row==2 && col==2)
          continue;
					// Calculate neighbour's coordinates in G //
				  // using modulus to satisfy boundary conditions //
        gridRowIdx = (row - 2 + momentRow + n) % n;
        gridColIdx = (col - 2 + momentCol + n) % n;

        weightFactor+= gpuG[gridRowIdx * n + gridColIdx] * gpuWeights[row*5+col];
      }
    }
		  // Update moment's atomic spin //
			// Set flag if a spin value transition has been done //
    if(weightFactor < 0.0001 && weightFactor > -0.0001)
    {
      gpuTempGrid[n*momentRow+momentCol] = gpuG[n*momentRow+momentCol];
    }else if(weightFactor > 0.00001)
    {
      gpuTempGrid[n*momentRow+momentCol] = 1;
			*flag = 1;
    }else
    {
      gpuTempGrid[n*momentRow+momentCol] = -1;
			*flag = 1;
    }
  }
}

void ising( int *G, double *w, int k, int n)
{
	GpuTimer timer;
	// Array to store weights in GPU memory //
  double *gpuWeights;
	// Array to store G in GPU memory //
	int *gpuG;
	// Array used for ising calculations in GPU //
	int *gpuTempGrid;
	// Variable storing the desired number of active blocks (grid size) //
	int GRID_SIZE;
	// Variable used for pointers swapping //
	int *gpuTempPtr;
	// Variables used to stop updating if the lattice remains the same //
	int flag;
	int *gpuFlag;
	// Start timing using Cuda Events //
	// Functions are written in gputimer.h //
	timer.Start();
	// Number of active blocks is calculated based on n-size //
	if(n % BLOCK_SIZE == 0)
		GRID_SIZE = n/BLOCK_SIZE;
	else
		GRID_SIZE = n/BLOCK_SIZE + 1;
	// cudaMalloc and cudaMemcpy used to allocate memory space //
	// and transfer data to the GPU //
	cudaMalloc(&gpuWeights, 25*sizeof(double));
	cudaMalloc(&gpuFlag, sizeof(int));
  cudaMemcpy(gpuWeights, w, 25*sizeof(double), cudaMemcpyHostToDevice);
	cudaMalloc(&gpuG, n*n*sizeof(int));
	cudaMemcpy(gpuG, G, n*n*sizeof(int), cudaMemcpyHostToDevice);
	cudaMalloc(&gpuTempGrid, n*n*sizeof(int));
	// Set Grid and Block dimensions //
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid(GRID_SIZE, GRID_SIZE);
	// Update k times //
	for(int i = 0; i < k; i++){
		flag=0;
		cudaMemcpy(gpuFlag, &flag, sizeof(int), cudaMemcpyHostToDevice);
		// Kernel function is executed by every thread //
		cudaKernel<<<dimGrid , dimBlock>>>(n, gpuWeights, gpuG, gpuTempGrid, gpuFlag);

		// Thread synchronization. Wait for every thread to be executed //
		// befor proceeding to the next iteration //
		cudaDeviceSynchronize();

		// Swap gpu_G and gpu_gTemp pointers for next iteration to avoid copying data on every iteration //
		gpuTempPtr = gpuG;
		gpuG = gpuTempGrid;
		gpuTempGrid = gpuTempPtr;

		cudaMemcpy(&flag, gpuFlag, sizeof(int), cudaMemcpyDeviceToHost);
		// Break if the lattice remained the same as before //
		// if(flag==0)
		// 	break;
	}

	// Transfer data back to CPU memory and store them in G //
  cudaMemcpy(G, gpuG, n*n*sizeof(int), cudaMemcpyDeviceToHost);
	// Stop timing //
	timer.Stop();
	printf("[V1] n: %d\tk: %d\tExecution Time(ms): %g\tGRID_SIZE: %d\tBLOCK_SIZE: %d\n",n,k,timer.Elapsed(),GRID_SIZE,BLOCK_SIZE);
	// Free allocated memory from the GPU //
	cudaFree(gpuG);
	cudaFree(gpuTempGrid);
	cudaFree(gpuWeights);
}
