//https://www.cs.uaf.edu/courses/cs441/notes/avx/
//gcc -O2 parallel_matrix_multiply.c -mavx -mfma && ./a.out
#include <stdint.h>
#include <time.h>
#include <sched.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <string.h>
#include <immintrin.h>
#include <pthread.h>
#include <unistd.h>
#include <stdatomic.h>

#define N 1024
#define BLOCK 8
#define FAST
#define REGISTERS
// #define DEBUG


static float A[N*N] __attribute__ ((aligned (32))),
             B[N*N] __attribute__ ((aligned (32))),
             C[N*N] __attribute__ ((aligned (32))),
          eval[N*N] __attribute__ ((aligned (32)));

__m256* Cm = (__m256*)C;

uint64_t nanos(){
  struct timespec start;
  clock_gettime(CLOCK_MONOTONIC_RAW, &start);
  return (uint64_t)start.tv_sec * 1e+9 + (uint64_t)start.tv_nsec;
}

void matmul(){
#ifndef FAST
  float cur = 0;
  for (int y = 0; y < N; ++y){
    for (int x = 0; x < N; ++x){
      cur = 0.;
      for (int k = 0; k < N; ++k){
        cur += A[y * N + k] * B[k * N + x];
      }
      C[y * N + x] = cur;
    }
  }
#else
  #ifndef REGISTERS
    float cur; 
    for (int yb = 0; yb < N; yb += BLOCK){
      for (int xb = 0; xb < N; xb += BLOCK){
        for (int k =0; k < N; ++k){
          for (int y = yb; y < yb + BLOCK; ++y){
            for (int x = xb; x < xb + BLOCK; ++x){
              // _mm256_store_ps(&cur, _mm256_mul_ps(Am, _mm256_load_ps(&B[k * N + x])));
              C[y*N + x] += A[y * N + k] * B[k * N + x];
            }
          }
        }
      }
    }
  #else
  // 32 GFLOPS
    __m256* Bm = (__m256*)B; // N * N -> N * N / 8
    for (int yb = 0; yb < N; yb+=BLOCK){
      for (int xb = 0; xb < N; xb+=BLOCK){

        __m256 acc[BLOCK][BLOCK / 8] = {};
        for (int k = 0; k < N; ++k){
          for (int y = 0; y < BLOCK; ++y){
            __m256 cur = _mm256_broadcast_ss(&A[(yb + y) * N + k]);
            for (int x = 0; x < BLOCK; x+=8){
              // Cm[((yb + y) * N  + xb + x) / 8] = _mm256_fmadd_ps(cur, Bm[(k*N + xb + x) / 8],
              //                                                         Cm[((yb + y) * N  + xb + x) / 8]);
              acc[y][x / 8] = _mm256_fmadd_ps(cur, Bm[(k*N + xb + x) / 8], acc[y][x/ 8]);
            }
          }
        }
      
        for (int cy = 0; cy < BLOCK; ++cy){
          for (int cx= 0; cx < BLOCK / 8; ++cx){
            Cm[((yb + cy) * N + xb + cx*8)/8] = acc[cy][cx];
          }
        }

      }
    }
    

  #endif

#endif
}


int main(){
  memset(C, 0, N*N);
  FILE* data = fopen("data", "rb");
  if (data == NULL){
    printf("please regenerate python file\n");
    return 1;
  }
  if (N % BLOCK != 0){
    printf("BAD BLOCK SIZE");
    return -1;
  }
  fread(A, 1, sizeof(float)*N*N, data);
  fread(B, 1, sizeof(float)*N*N, data);
  fread(eval, 1, sizeof(float)*N*N, data);
  fclose(data);
#ifndef DEBUG
  uint64_t s = nanos();
  matmul();
  uint64_t e = nanos();
#ifdef REGISTERS
  for (int i = 0; i < N / 8; ++i){
    for (int j = 0; j < 8; ++j){
      C[8*i + j] = Cm[i][j];
    }
  }
#endif
  printf("%f GFLOPS\n", 2.*N*N*N / (e - s));
  for (int i = 0; i < N; ++i){
    for (int j = 0; j < N; ++j){
      if (fabs(C[i * N + j] - eval[i * N + j]) > 1e-3){
        printf("MISMATCH at [%d, %d] : %f != %f", i, j, C[i * N + j], eval[i * N + j]);
        return 2;
      }
    }
  }
  printf("match");
#else
  matmul();
  printf("%f %f %f", C[0], C[1], C[2]);
#endif
}

