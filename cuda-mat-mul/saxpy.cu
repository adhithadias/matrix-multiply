#include <cmath>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static void HandleError(cudaError_t err, const char *file, int line) {
  if (err != cudaSuccess) {
    printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
    exit(EXIT_FAILURE);
  }
}
#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__))

#define HANDLE_NULL(a)                                                         \
  {                                                                            \
    if (a == NULL) {                                                           \
      printf("Host memory failed in %s at line %d\n", __FILE__, __LINE__);     \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  }

__global__ void saxpy(int *x, int *y, int alpha, size_t N) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < N) {
    y[i] = alpha * y[i] + x[i];
  }
}

void initialize_list(int *x, int N) {
  for (int i = 0; i < N; i++) {
    x[i] = rand();
  }
}

int main(void) {
  int N = std::pow(10, 7);
  int alpha = 2;

  int *x, *y;
  x = (int *)(malloc(N * sizeof(int)));
  y = (int *)(malloc(N * sizeof(int)));

  srand(time(NULL));
  initialize_list(x, N);
  initialize_list(y, N);

  int *d_x, *d_y;
  HANDLE_ERROR( cudaMalloc((void **)&d_x, N * sizeof(int)) );
  cudaMalloc((void **)&d_y, N * sizeof(int));

  cudaMemcpy(d_x, x, N * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, y, N * sizeof(int), cudaMemcpyHostToDevice);

  saxpy<<<(N + 255) / 256, 256>>>(d_x, d_y, alpha, N);

  int *c;
  c = (int *)(malloc(N * sizeof(int)));

  cudaMemcpy(c, d_y, N * sizeof(int), cudaMemcpyDeviceToHost);

  printf("[");
  for (int i = 0; i < N; i++) {

    if (i < 10) {
      printf("%d ", c[i]);
    }

    if (y[i] * alpha + x[i] != c[i]) {
      printf("YOU SCREWED UP!");
    }
  }
  printf(" ... ]");

  cudaFree(d_x);
  return 0;
}
