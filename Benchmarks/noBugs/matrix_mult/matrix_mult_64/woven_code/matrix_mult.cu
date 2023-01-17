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

__global__ void multiMatrix(int *d_a, int *d_b, int *d_c, int N, unsigned char * traceArray, int numberOfThreads) {
   int cont = 0;
   int map = (blockIdx.x  + gridDim.x  * blockIdx.y) * (blockDim.x * blockDim.y) + (threadIdx.x + blockDim.x * threadIdx.y);
   registerTrace(traceArray, numberOfThreads, map, cont++, 1);
   int idx = threadIdx.x + blockDim.x * blockIdx.x;
   int idy = threadIdx.y + blockDim.y * blockIdx.y;
   int pos = idx + idy * N;
   int temp_result = 0;
   int posa, posb;
   static __shared__ int s_a[16][16];
   static __shared__ int s_b[16][16];
   int tile_idx;
   for(/* node 2 */ tile_idx = 0; /* node 3 */ tile_idx < gridDim.x; /* node 24 */ tile_idx++) {
      registerTrace(traceArray, numberOfThreads, map, cont++, 4);
      posa = idy * N + (tile_idx * 16 + threadIdx.x);
      posb = (tile_idx * 16 + threadIdx.y) * N + idx;
      /* node 5 */
      if(posa < N * N) {
         registerTrace(traceArray, numberOfThreads, map, cont++, 6);
         s_a[threadIdx.y][threadIdx.x] = d_a[posa];
      }
      else {
         registerTrace(traceArray, numberOfThreads, map, cont++, 7);
         s_a[threadIdx.y][threadIdx.x] = 0;
      }
      /* node 8 */
      /* node 9 */
      if(posb < N * N) {
         registerTrace(traceArray, numberOfThreads, map, cont++, 10);
         s_b[threadIdx.y][threadIdx.x] = d_b[posb];
      }
      else {
         registerTrace(traceArray, numberOfThreads, map, cont++, 11);
         s_b[threadIdx.y][threadIdx.x] = 0;
      }
      /* node 12 */
      /* node 13 sync */
      __syncthreads();
      /* node 14 */
      if(idx < N && idy < N) {
         registerTrace(traceArray, numberOfThreads, map, cont++, 15);
         int i;
         for(/* node 16 */ i = 0; /* node 17 */ i < 16; /* node 21 */ i++) {
            /* node 18 */
            if(s_a[threadIdx.y][i] != 0 && s_b[i][threadIdx.x] != 0) {
               registerTrace(traceArray, numberOfThreads, map, cont++, 19);
               temp_result += s_a[threadIdx.y][i] * s_b[i][threadIdx.x];
            }
            registerTrace(traceArray, numberOfThreads, map, cont++, 20);
         }
      }
      registerTrace(traceArray, numberOfThreads, map, cont++, 22);
      /* node 23 sync */
      __syncthreads();
   }
   registerTrace(traceArray, numberOfThreads, map, cont++, 25);
   __syncthreads();
   /* node 26 */
   if(idx < N && idy < N) {
      registerTrace(traceArray, numberOfThreads, map, cont++, 27);
      d_c[pos] = temp_result;
   }
   registerTrace(traceArray, numberOfThreads, map, cont++, 28);
}

int main(int argc, char *argv[]) {
   FILE *fp;
   FILE *fpHost = fopen("trace.Host.p0", "w");
   int N = atoi(argv[1]);
   registerTraceHost(fpHost, 1);
   size_t size = N * N * sizeof(int);
   int num_thread, num_block;
   int *h_a, *h_b, *h_c;
   h_a = (int *) malloc(size);
   h_b = (int *) malloc(size);
   h_c = (int *) malloc(size);
   int *d_a, *d_b, *d_c;
   /* node 2 */
   cudaMalloc(&d_a, size);
   /* node 3 */
   cudaMalloc(&d_b, size);
   /* node 4 */
   cudaMalloc(&d_c, size);
   int i;
   for(/* node 5 */ i = 0; /* node 6 */ i < N * N; /* node 8 */ i++) {
      scanf("%d", &h_a[i]);
      registerTraceHost(fpHost, 7);
   }
   for(/* node 9 */ i = 0; /* node 10 */ i < N * N; /* node 12 */ i++) {
      scanf("%d", &h_b[i]);
      registerTraceHost(fpHost, 11);
   }
   registerTraceHost(fpHost, 13);
   cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
   /* node 14 */
   cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);
   cudaMemset(d_c, 0, size);
   num_block = 4;
   num_thread = 16;
   dim3 gridsize(num_block, num_block, 1);
   dim3 blocksize(num_thread, num_thread, 1);
   int numberOfThreads0 = gridsize.x * gridsize.y * gridsize.z * blocksize.x * blocksize.y * blocksize.z;
   unsigned char *instTrace0;
   cudaMallocManaged(&instTrace0, numberOfThreads0 * 2000 * sizeof(unsigned char));
   /* node 15 */
   multiMatrix<<<gridsize, blocksize>>>(d_a, d_b, d_c, N, instTrace0, numberOfThreads0);
   /* node 16 */
   cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);
   for(/* node 17 */ i = 0; /* node 18 */ i < N; /* node 24 */ i++) {
      int j;
      registerTraceHost(fpHost, 19);
      for(registerTraceHost(fpHost, 20), j = 0; /* node 21 */ j < N; /* node 23 */ j++) {
         printf("%d ", h_c[i * N + j]);
         registerTraceHost(fpHost, 22);
      }
      printf("\n");
   }
   registerTraceHost(fpHost, 25);
   cudaFree(d_a);
   /* node 26 */
   cudaFree(d_b);
   /* node 27 */
   cudaFree(d_c);
   free(h_a);
   free(h_b);
   free(h_c);
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
