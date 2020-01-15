#include "../inc/ising.h"
#include <sys/time.h>
#include <string.h>

void ising(int *G, double *w, int k, int n)
{
  struct timeval start, end;
  // Start timing //
  gettimeofday(&start, NULL);
  // Variable storing the total neighbourhood influence //
  double weightFactor;
  double time;
  int gridRowIdx, gridColIdx;
  // Array used for swapping //
  int *tempGrid = (int *)malloc(sizeof(int)*n*n);
  int *tempPtr;
  // Flag used in order to stop if the lattice remains the same //
  int flag = 0;
  // Update lattice k times //
  for(int iteration=0; iteration<k; iteration++)
  {
    flag = 0;
    // Iterate through G //
    for(int j=0; j<n; j++)
    {
      for(int i=0; i<n; i++)
      {
        weightFactor = 0.0;
        // Read 24 neighbours of every moment and calculate their total influence //
        for(int wCol=0; wCol<5; wCol++)
        {
          for(int wRow=0; wRow<5; wRow++)
          {
            if(wRow==2 && wCol==2)
              continue;
            // Calculate neighbour's coordinates in G using //
    				// modulus to remain in bounds //
            gridColIdx = (wCol - 2 + j + n) % n;
            gridRowIdx = (wRow - 2 + i + n) % n;

            weightFactor+= G[gridColIdx * n + gridRowIdx] * w[wCol*5+wRow];
          }
        }
        // Update moment's atomic spin //
        if(weightFactor < 0.0001 && weightFactor > -0.0001)
        {
          tempGrid[n*j+i] = G[n*j+i];
        }else if(weightFactor > 0.00001)
        {
          tempGrid[n*j+i] = 1;
          flag=1;
        }else
        {
          tempGrid[n*j+i] = -1;
          flag=1;
        }
      }
    }
    // Swap pointers //
    tempPtr = G;
    G = tempGrid;
    tempGrid = tempPtr;
    // if(flag!=1)
    //   break;
  }
  // If k%2==1 the result is stored in G array //
  if(k % 2 == 1)
    memcpy(tempGrid, G, n*n*sizeof(int));

  // Stop timing //
  gettimeofday(&end, NULL);
  time = (double)((end.tv_usec - start.tv_usec)/1.0e6 + end.tv_sec - start.tv_sec);
  printf("Sequential\nn: %d\nk: %d\nExecution time(sec): %f\n",n,k,time);
}
