#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define NUMROWS 8
#define NUMCOLS 8
#define idx(u, y, x) (u[y * NUMCOLS + x])

float* newArray(int rows, int cols) {
  float* a = (float*)malloc(NUMROWS * NUMCOLS * sizeof(float));
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      idx(a, i, j) = i*cols+j;
    }
  }
  return a;
}

void printArray(float* a, int rows, int cols) {
   for (int i=0; i<rows; i++) {
      for (int j=0; j<cols; j++) {
         printf("%.2f ", *(a + i*cols + j));
      }
      printf("\n");
   }
   printf("\n\n\n");
}


void matmul_host(float* a, float* b, float* c, int r1, int c1, int c2) {
  for (int i = 0; i < r1; i++) {
    for (int j = 0; j < c2; j++) {
      float comp = 0.;
      for (int k = 0; k < c1; k++) {
        comp += a[i*c1+k] * b[k*c1+j];
      }
      idx(c, i, j) = comp;
    }
  }
}

__global__
void matmul(float* a, float* b, float* c, int r1, int c1, int c2) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int j = blockDim.y * blockIdx.y + threadIdx.y;

  if (i<r1 && j<c2) {
    float comp = 0.;
    for (int k = 0; k < c1; k++) {
      comp += a[i*c1+k] * b[k*c1+j];
    }
    c[0] = 100;
    c[1] = 200;
  }
}

__global__ void gpu_matrix_mult(float *a,float *b, float *c, int m, int n, int k)
{ 
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0;
    if( col < k && row < m) 
    {
        for(int i = 0; i < n; i++) 
        {
            sum += a[row * n + i] * b[i * k + col];
        }
        c[row * k + col] = sum;
    }
} 

int main(int argc, char** args) {

  float* a = newArray(NUMROWS, NUMCOLS);
  float* b = newArray(NUMROWS, NUMCOLS);
  float* c = (float *) malloc(NUMROWS*NUMCOLS*sizeof(float));

  float *d_x, *d_y, *d_z;
  // cudaMallocManaged is used to allocate unifies memory 
  // accessible through both the CPU and GPU
	cudaMalloc((void **)&d_x, NUMROWS*NUMCOLS*sizeof(float));
  cudaMalloc((void **)&d_y, NUMROWS*NUMCOLS*sizeof(float));
  cudaMalloc((void **)&d_z, NUMROWS*NUMCOLS*sizeof(float));

  clock_t begin = clock();

  cudaMemcpy(d_x, a, NUMROWS*NUMCOLS*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, b, NUMROWS*NUMCOLS*sizeof(float), cudaMemcpyHostToDevice);
  int threads = 32;
  dim3 dim_grid((NUMROWS+31)/threads, (NUMCOLS+31)/threads, 1);
  dim3 dim_block(threads, threads, 1);
  gpu_matrix_mult<<<dim_grid, dim_block>>>(d_x, d_y, d_z, NUMROWS, NUMCOLS, NUMCOLS);
  cudaMemcpy(c, d_z, NUMROWS*NUMCOLS*sizeof(float), cudaMemcpyDeviceToHost);

  clock_t end = clock();
  double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
  printf("Elapsed: %f seconds\n", time_spent);

  printArray(c, NUMROWS, NUMCOLS);
  
  begin = clock();

  matmul_host(a, b, c, NUMROWS, NUMCOLS, NUMCOLS);
  
  end = clock();
  
  time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
  printf("Elapsed: %f seconds\n", time_spent);

  printArray(c, NUMROWS, NUMCOLS);

    // Free memory
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_y);

    free(a); free(b); free(c);
}
