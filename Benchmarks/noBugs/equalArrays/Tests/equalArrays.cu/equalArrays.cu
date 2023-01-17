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

__global__ void verifyArray(int *d_a, int *d_b, bool *d_equal, int N, unsigned char * traceArray, int numberOfThreads) {
   int cont = 0;
   int map = (blockIdx.x  + gridDim.x  * blockIdx.y) * (blockDim.x * blockDim.y) + (threadIdx.x + blockDim.x * threadIdx.y);
   registerTrace(traceArray, numberOfThreads, map, cont++, 1);
   int pos = threadIdx.x + blockDim.x * blockIdx.x;
   /* node 2 */
   if(d_a[pos] != d_b[pos] && pos < N) { registerTrace(traceArray, numberOfThreads, map, cont++, 3); *d_equal = false; }
   registerTrace(traceArray, numberOfThreads, map, cont++, 4);
}

int main(int argc, char *argv[]) {
   FILE *fp;
   FILE *fpHost = fopen("trace.Host.p0", "w");
   int N = atoi(argv[1]);
   registerTraceHost(fpHost, 1);
   size_t size = N * sizeof(int);
   bool *h_equal = (bool *) malloc(sizeof(bool));
   *h_equal = true;
   int *h_a, *h_b;
   h_a = (int *) malloc(size);
   h_b = (int *) malloc(size);
   int *d_a, *d_b;
   bool *d_equal;
   /* node 2 */
   cudaMalloc(&d_a, size);
   /* node 3 */
   cudaMalloc(&d_b, size);
   /* node 4 */
   cudaMalloc(&d_equal, sizeof(bool));
   int i;
   for(/* node 5 */ i = 0; /* node 6 */ i < N; /* node 8 */ i++) {
      scanf("%d", &h_a[i]);
      registerTraceHost(fpHost, 7);
   }
   for(registerTraceHost(fpHost, 9), i = 0; /* node 10 */ i < N; /* node 12 */ i++) {
      scanf("%d", &h_b[i]);
      registerTraceHost(fpHost, 11);
   }
   int num_block = 32;
   int num_thread = 256;
   registerTraceHost(fpHost, 13);
   cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
   /* node 14 */
   cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);
   dim3 gridsize(num_block, 1, 1);
   dim3 blocksize(num_thread, 1, 1);
   /* node 15 */
   cudaMemcpy(d_equal, h_equal, sizeof(bool), cudaMemcpyHostToDevice);
   int numberOfThreads0 = gridsize.x * gridsize.y * gridsize.z * blocksize.x * blocksize.y * blocksize.z;
   unsigned char *instTrace0;
   cudaMallocManaged(&instTrace0, numberOfThreads0 * 2000 * sizeof(unsigned char));
   /* node 16 */
   verifyArray<<<gridsize, blocksize>>>(d_a, d_b, d_equal, N, instTrace0, numberOfThreads0);
   /* node 17 */
   cudaMemcpy(h_equal, d_equal, sizeof(bool), cudaMemcpyDeviceToHost);
   printf("%s\n", *h_equal ? "true" : "false");
   /* node 18 */
   cudaFree(d_a);
   /* node 19 */
   cudaFree(d_b);
   registerTraceHost(fpHost, 20);
   cudaFree(d_equal);
   free(h_a);
   free(h_b);
   free(h_equal);
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
