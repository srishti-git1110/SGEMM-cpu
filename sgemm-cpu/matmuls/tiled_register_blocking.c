#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <arm_neon.h>

#ifndef N
#define N 4096
#endif

#define IR 4
#define JR 4
#define TILE_I 64
#define TILE_J 64
#define TILE_K 128

double timeDiff(struct timeval *start, struct timeval *end) {
    double start_sec = start->tv_sec + (start->tv_usec / 1000000.0);
    double end_sec = end->tv_sec + (end->tv_usec / 1000000.0);
    return end_sec - start_sec;
}

float A[N][N], B[N][N], C[N][N];

int main(int argc, char *argv[]) {
    (void)argc;
    (void)argv;

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[i][j] = (float)(i + j) / (float)RAND_MAX;
            B[i][j] = (float)(i * j) / (float)RAND_MAX;
            C[i][j] = 0.0;
        }
    }

    struct timeval start;
    gettimeofday(&start, NULL);

    for (int i_tile = 0; i_tile < N; i_tile += TILE_I) {
        int iend = (i_tile + TILE_I < N) ? i_tile + TILE_I : N;

        for (int j_tile = 0; j_tile < N; j_tile += TILE_J) {
            int jend = (j_tile + TILE_J < N) ? j_tile + TILE_J : N;

            for (int k_tile = 0; k_tile < N; k_tile += TILE_K) {
                int kend = (k_tile + TILE_K < N) ? k_tile + TILE_K : N;

                for (int i = i_tile; i < iend; i += IR) {
                    for (int j = j_tile; j < jend; j += JR) {

                        float32x4_t c0 = vld1q_f32(&C[i+0][j]);
                        float32x4_t c1 = vld1q_f32(&C[i+1][j]);
                        float32x4_t c2 = vld1q_f32(&C[i+2][j]);
                        float32x4_t c3 = vld1q_f32(&C[i+3][j]);

                        for (int k = k_tile; k < kend; k++) {
                            float32x4_t b = vld1q_f32(&B[k][j]);

                            c0 = vfmaq_n_f32(c0, b, A[i+0][k]);
                            c1 = vfmaq_n_f32(c1, b, A[i+1][k]);
                            c2 = vfmaq_n_f32(c2, b, A[i+2][k]);
                            c3 = vfmaq_n_f32(c3, b, A[i+3][k]);
                        }

                        vst1q_f32(&C[i+0][j], c0);
                        vst1q_f32(&C[i+1][j], c1);
                        vst1q_f32(&C[i+2][j], c2);
                        vst1q_f32(&C[i+3][j], c3);
                    }
                }
            }
        }
    }

    struct timeval end;
    gettimeofday(&end, NULL);

    printf("time taken for register-blocked matmul (%dx%d C tile): %0.8lf\n",
           IR, JR, timeDiff(&start, &end));

    double checksum = 0.0;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            checksum += C[i][j];
        }
    }
    printf("sum of C: %0.8lf\n", checksum);
    return 0;
}
