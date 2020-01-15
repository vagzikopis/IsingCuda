#include "ising.h"
#include "gputimer.h"
int main(int argc, char* argv[]){
	if(argc < 2 )
  {
    printf("Please enter 2 arguments.\nn(size)\nk(iterations)\n");
  }
	// Initialise parameters //
	int n = atoi(argv[1]);	int k = atoi(argv[2]); int isValid=1;

	// struct timeval start, end;
	int *G = (int*)malloc(n*n*sizeof(int));
  double weights[] = {0.004, 0.016, 0.026, 0.016, 0.004,
	  	0.016, 0.071, 0.117, 0.071, 0.016,
			0.026, 0.117, 0, 0.117, 0.026,
			0.016, 0.071, 0.117, 0.071, 0.016,
			0.004, 0.016, 0.026, 0.016, 0.004};
	for (int i=0; i<n*n; i++)
	        G[i] = ((rand() % 2) * 2) - 1;
	ising(G, weights, k, n);
  // FILE *fptr = fopen("conf-init.bin","rb");
  // if (fptr == NULL){
	// 		// File pointer error //
	// 		exit(1);
  // }
  // fread(G, sizeof(int), n*n, fptr);
  // fclose(fptr);
  // // Initialise weights array //
	// gettimeofday(&start, NULL);
  // ising(G, weights, k, n);
	// gettimeofday(&end, NULL);
	// printf("Ising time: %f\n", (end.tv_usec-start.tv_usec)/1.0e6 + end.tv_sec - start.tv_sec);
	// // Test function validity //
  // int *validResult = (int*)malloc(n*n*sizeof(int));
	// fptr = fopen("conf-1.bin","rb");
	// fread(validResult, sizeof(int), n*n, fptr);
	// fclose(fptr);
  // for(int v = 0; v < n*n; v++)
  // {
  //   if(validResult[v] != G[v])
  //   {
  //     isValid = 0;
  //     break;
  //   }
  // }
  // if(isValid)
  //   printf("Passed k=1\n");
  // else
  //   printf("Failed k=1\n");
	//
	// // 4 iterations in total //
	// ising(G, weights, 3, n);
	// fptr = fopen("conf-4.bin","rb");
	// fread(validResult, sizeof(int), n*n, fptr);
	// fclose(fptr);
  // for(int v = 0; v < n*n; v++)
  // {
  //   if(validResult[v] != G[v])
  //   {
  //     isValid = 0;
  //     break;
  //   }
  // }
  // if(isValid)
  //   printf("Passed k=4\n");
  // else
  //   printf("Failed k=4\n");
	// // 11 iterations in total //
	// ising(G, weights, 7, n);
	// fptr = fopen("conf-11.bin","rb");
	// fread(validResult, sizeof(int), n*n, fptr);
	// fclose(fptr);
  // for(int v = 0; v < n*n; v++)
  // {
  //   if(validResult[v] != G[v])
  //   {
  //     isValid = 0;
  //     break;
  //   }
  // }
  // if(isValid)
  //   printf("Passed k=11\n");
  // else
  //   printf("Failed k=11\n");
	//
  // return 0;
}
