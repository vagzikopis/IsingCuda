#include "../inc/ising.h"

int main(int argc, char* argv[]){

	int n = 517;	int k = 1; int isValid=1;

	struct timeval start, end;
	int *G = (int*)malloc(n*n*sizeof(int));
  double weights[] = {0.004, 0.016, 0.026, 0.016, 0.004,
	  	0.016, 0.071, 0.117, 0.071, 0.016,
			0.026, 0.117, 0, 0.117, 0.026,
			0.016, 0.071, 0.117, 0.071, 0.016,
			0.004, 0.016, 0.026, 0.016, 0.004};

  FILE *fptr = fopen("../testdata/conf-init.bin","rb");
  if (fptr == NULL){
			// File pointer error //
			exit(1);
  }
  fread(G, sizeof(int), n*n, fptr);
  fclose(fptr);
  ising(G, weights, k, n);

	// Test function validity //
  int *validResult = (int*)malloc(n*n*sizeof(int));
	fptr = fopen("../testdata/conf-1.bin","rb");
	fread(validResult, sizeof(int), n*n, fptr);
	fclose(fptr);
  for(int v = 0; v < n*n; v++)
  {
    if(validResult[v] != G[v])
    {
      isValid = 0;
      break;
    }
  }
  if(isValid)
    printf("Passed k=1\n");
  else
    printf("Failed k=1\n");

	// 4 iterations in total //
	ising(G, weights, 3, n);
	fptr = fopen("../testdata/conf-4.bin","rb");
	fread(validResult, sizeof(int), n*n, fptr);
	fclose(fptr);
  for(int v = 0; v < n*n; v++)
  {
    if(validResult[v] != G[v])
    {
      isValid = 0;
      break;
    }
  }
  if(isValid)
    printf("Passed k=4\n");
  else
    printf("Failed k=4\n");
	// 11 iterations in total //
	ising(G, weights, 7, n);
	fptr = fopen("../testdata/conf-11.bin","rb");
	fread(validResult, sizeof(int), n*n, fptr);
	fclose(fptr);
  for(int v = 0; v < n*n; v++)
  {
    if(validResult[v] != G[v])
    {
      isValid = 0;
      break;
    }
  }
  if(isValid)
    printf("Passed k=11\n");
  else
    printf("Failed k=11\n");

  return 0;

	}
