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

__global__ void diagonal(int *d_a, int *d_b, int N, int mode, unsigned char * traceArray, int numberOfThreads) {
   int cont = 0;
   int map = (blockIdx.x  + gridDim.x  * blockIdx.y) * (blockDim.x * blockDim.y) + (threadIdx.x + blockDim.x * threadIdx.y);
   registerTrace(traceArray, numberOfThreads, map, cont++, 1);
   int idx = threadIdx.x + blockDim.x * blockIdx.y; // bug
   int idy = threadIdx.y + blockDim.y * blockIdx.y;
   int pos = idx + idy * N;
   /* node 2 */
   if(mode == 0) { // main diagonal
      registerTrace(traceArray, numberOfThreads, map, cont++, 3);
      if(idx < N && idy < N) {
         registerTrace(traceArray, numberOfThreads, map, cont++, 4);
         if(idx == idy) { registerTrace(traceArray, numberOfThreads, map, cont++, 5); d_b[pos] = d_a[pos]; }
         else {
            registerTrace(traceArray, numberOfThreads, map, cont++, 6);
            d_b[pos] = 0;
         }
         /* node 7 */
      }
      registerTrace(traceArray, numberOfThreads, map, cont++, 8);
   }
   registerTrace(traceArray, numberOfThreads, map, cont++, 9);
   /* node 10 */
   if(mode == 1) { // secondary diagonal
      registerTrace(traceArray, numberOfThreads, map, cont++, 11);
      if(idx < N && idy < N) {
         registerTrace(traceArray, numberOfThreads, map, cont++, 12);
         if(idx + idy == N - 1) { registerTrace(traceArray, numberOfThreads, map, cont++, 13); d_b[pos] = d_a[pos]; }
         else {
            registerTrace(traceArray, numberOfThreads, map, cont++, 14);
            d_b[pos] = 0;
         }
         /* node 15 */
      }
      registerTrace(traceArray, numberOfThreads, map, cont++, 16);
   }
   registerTrace(traceArray, numberOfThreads, map, cont++, 17);
   /* node 18 */
   if(mode == 2) { // inverse main diagonal
      registerTrace(traceArray, numberOfThreads, map, cont++, 19);
      int posb = N * N - 1 - pos;
      /* node 20 */
      if(idx < N && idy < N) {
         registerTrace(traceArray, numberOfThreads, map, cont++, 21);
         if(idx == idy) { registerTrace(traceArray, numberOfThreads, map, cont++, 22); d_b[pos] = d_a[posb]; }
         else {
            registerTrace(traceArray, numberOfThreads, map, cont++, 23);
            d_b[pos] = 0;
         }
         /* node 24 */
      }
      registerTrace(traceArray, numberOfThreads, map, cont++, 25);
   }
   registerTrace(traceArray, numberOfThreads, map, cont++, 26);
   /* node 27 */
   if(mode == 3) { // inverse secondary diagonal
      registerTrace(traceArray, numberOfThreads, map, cont++, 28);
      int posb = idx * N + idy;
      /* node 29 */
      if(idx < N && idy < N) {
         registerTrace(traceArray, numberOfThreads, map, cont++, 30);
         if(idx + idy == N - 1) { registerTrace(traceArray, numberOfThreads, map, cont++, 31); d_b[pos] = d_a[posb]; }
         else {
            registerTrace(traceArray, numberOfThreads, map, cont++, 32);
            d_b[pos] = 0;
         }
         /* node 33 */
      }
      registerTrace(traceArray, numberOfThreads, map, cont++, 34);
   }
   registerTrace(traceArray, numberOfThreads, map, cont++, 35);
}

int main(int argc, char *argv[]) {
   FILE *fp;
   FILE *fpHost = fopen("trace.Host.p0", "w");
   int N = atoi(argv[1]);
   registerTraceHost(fpHost, 1);
   int mode = atoi(argv[2]);
   size_t size = N * N * sizeof(int);
   int num_thread, num_block;
   int i, j;
   int *h_a, *h_b;
   h_a = (int *) malloc(size);
   h_b = (int *) malloc(size);
   int *d_a, *d_b;
   /* node 2 */
   cudaMalloc(&d_a, size);
   /* node 3 */
   cudaMalloc(&d_b, size);
   for(/* node 4 */ i = 0; /* node 5 */ i < N * N; /* node 7 */ i++) {
      scanf("%d", &h_a[i]);
      registerTraceHost(fpHost, 6);
   }
   registerTraceHost(fpHost, 8);
   cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
   cudaMemset(d_b, 0, size);
   num_block = 2;
   num_thread = 32;
   printf("Blocks: %d    Threads: %d  \n", num_block, num_thread);
   dim3 gridsize(num_block, num_block, 1);
   dim3 blocksize(num_thread, num_thread, 1);
   int numberOfThreads0 = gridsize.x * gridsize.y * gridsize.z * blocksize.x * blocksize.y * blocksize.z;
   unsigned char *instTrace0;
   cudaMallocManaged(&instTrace0, numberOfThreads0 * 2000 * sizeof(unsigned char));
   /* node 9 */
   diagonal<<<gridsize, blocksize>>>(d_a, d_b, N, mode, instTrace0, numberOfThreads0);
   /* node 10 */
   cudaMemcpy(h_b, d_b, size, cudaMemcpyDeviceToHost);
   for(/* node 11 */ i = 0; /* node 12 */ i < N; registerTraceHost(fpHost, 17), i++) {
      for(registerTraceHost(fpHost, 13), j = 0; /* node 14 */ j < N; /* node 16 */ j++) {
         printf("%d ", h_b[i * N + j]);
         registerTraceHost(fpHost, 15);
      }
      printf("\n");
   }
   registerTraceHost(fpHost, 18);
   cudaFree(d_a);
   registerTraceHost(fpHost, 19);
   cudaFree(d_b);
   free(h_a);
   free(h_b);
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
