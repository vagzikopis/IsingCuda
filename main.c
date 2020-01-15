#include "/inc/ising.h"

int main(int argc, char* argv[])
{
	if(argc < 2 )
  {
    printf("Please enter 2 arguments.\nn(size)\nk(iterations)\n");
  }
//	Initialise parameters //
	int n = atoi(argv[1]);	int k = atoi(argv[2]); int isValid=1;

	struct timeval start, end;
	int *G = (int*)malloc(n*n*sizeof(int));
  double weights[] = {0.004, 0.016, 0.026, 0.016, 0.004,
	  	0.016, 0.071, 0.117, 0.071, 0.016,
			0.026, 0.117, 0, 0.117, 0.026,
			0.016, 0.071, 0.117, 0.071, 0.016,
			0.004, 0.016, 0.026, 0.016, 0.004};
	for (int i=0; i<n*n; i++)
	        G[i] = ((rand() % 2) * 2) - 1;
	ising(G, weights, k, n);
	return 0;
}
