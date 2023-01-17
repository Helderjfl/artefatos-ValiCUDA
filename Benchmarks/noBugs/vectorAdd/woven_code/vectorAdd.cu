#include <stdio.h>
__device__ void registerTrace(unsigned char *traceArray, int width, int id, int cont, unsigned char node)
{
	traceArray[cont * width + id] = node;
}

unsigned char history[2] = {0, 0};
__host__ void registerTraceHost(FILE *fp, unsigned char node)
{
	if (history[0] != node || history[1] != node)
		fprintf(fp, "%d-0\t", node);
	history[0] = history[1];
	history[1] = node;
}

__global__ void vectorAdd(int *d_a, int *d_b, int *d_c, unsigned char * traceArray, int numberOfThreads) {
   int cont = 0;
   int map = (blockIdx.x  + gridDim.x  * blockIdx.y) * (blockDim.x * blockDim.y) + (threadIdx.x + blockDim.x * threadIdx.y);
   registerTrace(traceArray, numberOfThreads, map, cont++, 1);
   int id = threadIdx.x;
   d_c[id] = d_a[id] + d_b[id];
}

int main(int argc, char **argv) {
   FILE *fp;
   FILE *fpHost = fopen("trace.Host.p0", "w");
   int N = atoi(argv[1]);
   registerTraceHost(fpHost, 1);
   int i;
   int *a;
   int *b;
   int *c;
   size_t size = N * sizeof(int);
   a = (int *) malloc(size);
   b = (int *) malloc(size);
   c = (int *) malloc(size);
   for(/* node 2 */ i = 0; /* node 3 */ i < N; /* node 5 */ i++) {
      a[i] = b[i] = i;
      registerTraceHost(fpHost, 4);
   }
   int *d_a;
   int *d_b;
   int *d_c;
   registerTraceHost(fpHost, 6);
   cudaMalloc(&d_a, size);
   /* node 7 */
   cudaMalloc(&d_b, size);
   /* node 8 */
   cudaMalloc(&d_c, size);
   /* node 9 */
   cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
   /* node 10 */
   cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);
   int numberOfThreads0 = 1 * 1024;
   unsigned char *instTrace0;
   cudaMallocManaged(&instTrace0, numberOfThreads0 * 2000 * sizeof(unsigned char));
   /* node 11 */
   vectorAdd<<<1, 1024>>>(d_a, d_b, d_c, instTrace0, numberOfThreads0);
   /* node 12 */
   cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);
   for(/* node 13 */ i = 0; /* node 14 */ i < N; /* node 16 */ i++) {
      printf("%d ", c[i]);
      registerTraceHost(fpHost, 15);
   }
   cudaDeviceReset();
   registerTraceHost(fpHost, 17);
   cudaFree(d_a);
   /* node 18 */
   cudaFree(d_b);
   registerTraceHost(fpHost, 19);
   cudaFree(d_c);
   fclose(fpHost);
   fp = fopen("trace.Grid.p1", "w");
   for(int threadIndex = 0; threadIndex < numberOfThreads0; threadIndex++)
   {
   	for(int nodeTrace = 0; nodeTrace < 2000; nodeTrace++)
   	{
   		if(instTrace0[threadIndex + nodeTrace * numberOfThreads0] != 0){
   			if(nodeTrace >= 2){
   				if(instTrace0[threadIndex + nodeTrace * numberOfThreads0] != instTrace0[threadIndex + (nodeTrace-1) * numberOfThreads0] ||
   				instTrace0[threadIndex + nodeTrace * numberOfThreads0] != instTrace0[threadIndex + (nodeTrace-2) * numberOfThreads0])
   					fprintf(fp, "%d-1\t", instTrace0[threadIndex + nodeTrace * numberOfThreads0]);
   			}else
   				fprintf(fp, "%d-1\t", instTrace0[threadIndex + nodeTrace * numberOfThreads0]);
   		}
   	}
   	fprintf(fp, "\n");
   }
   fclose(fp);
   fp = fopen("commsize", "w");
   fprintf(fp, "2");
   fclose(fp);
}
