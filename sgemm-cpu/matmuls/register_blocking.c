#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <arm_neon.h>

#ifndef N
#define N 4096
#endif

#define IR 4
#define JR 4

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

    for (int i = 0; i < N; i += IR) {
        for (int j = 0; j < N; j += JR) {
            float c00 = C[i + 0][j + 0];
            float c01 = C[i + 0][j + 1];
            float c02 = C[i + 0][j + 2];
            float c03 = C[i + 0][j + 3];

            float c10 = C[i + 1][j + 0];
            float c11 = C[i + 1][j + 1];
            float c12 = C[i + 1][j + 2];
            float c13 = C[i + 1][j + 3];

            float c20 = C[i + 2][j + 0];
            float c21 = C[i + 2][j + 1];
            float c22 = C[i + 2][j + 2];
            float c23 = C[i + 2][j + 3];

            float c30 = C[i + 3][j + 0];
            float c31 = C[i + 3][j + 1];
            float c32 = C[i + 3][j + 2];
            float c33 = C[i + 3][j + 3];

            for (int k = 0; k < N; k++) {
                float a0 = A[i + 0][k];
                float a1 = A[i + 1][k];
                float a2 = A[i + 2][k];
                float a3 = A[i + 3][k];

                float b0 = B[k][j + 0];
                float b1 = B[k][j + 1];
                float b2 = B[k][j + 2];
                float b3 = B[k][j + 3];

                c00 += a0 * b0;
                c01 += a0 * b1;
                c02 += a0 * b2;
                c03 += a0 * b3;

                c10 += a1 * b0;
                c11 += a1 * b1;
                c12 += a1 * b2;
                c13 += a1 * b3;

                c20 += a2 * b0;
                c21 += a2 * b1;
                c22 += a2 * b2;
                c23 += a2 * b3;

                c30 += a3 * b0;
                c31 += a3 * b1;
                c32 += a3 * b2;
                c33 += a3 * b3;
            }

            C[i + 0][j + 0] = c00;
            C[i + 0][j + 1] = c01;
            C[i + 0][j + 2] = c02;
            C[i + 0][j + 3] = c03;

            C[i + 1][j + 0] = c10;
            C[i + 1][j + 1] = c11;
            C[i + 1][j + 2] = c12;
            C[i + 1][j + 3] = c13;

            C[i + 2][j + 0] = c20;
            C[i + 2][j + 1] = c21;
            C[i + 2][j + 2] = c22;
            C[i + 2][j + 3] = c23;

            C[i + 3][j + 0] = c30;
            C[i + 3][j + 1] = c31;
            C[i + 3][j + 2] = c32;
            C[i + 3][j + 3] = c33;
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
