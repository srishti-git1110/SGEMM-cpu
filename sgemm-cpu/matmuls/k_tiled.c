#include <stdio.h>
#include <sys/time.h>
#include <stdlib.h>

#define N 4096
#define TILE_SIZE 32

double timeDiff(struct timeval *start, struct timeval *end) {
    double start_sec = start->tv_sec + (start->tv_usec / 1000000.0);
    double end_sec = end->tv_sec + (end->tv_usec / 1000000.0);
    return end_sec - start_sec;
}

float A[N][N], B[N][N], C[N][N];

int main(int argc, char *argv[]) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[i][j] = (float)(i + j) / (float) RAND_MAX;
            B[i][j] = (float)(i * j) / (float) RAND_MAX;
            C[i][j] = 0.0;
        }
    }

    struct timeval start;
    gettimeofday(&start, NULL);

    for (int k_tile=0; k_tile<N; k_tile+=TILE_SIZE) {
        for (int i=0; i<N; i++) {
            int kend = (k_tile + TILE_SIZE > N) ? N : k_tile + TILE_SIZE; // check if this is expensive than an if
            for (int k=k_tile; k<kend; k++) {
                float a_ik = A[i][k];
                for (int j=0; j<N; j++) {
                    C[i][j] += a_ik * B[k][j];
                }
            }
        }
    }

    struct timeval end;
    gettimeofday(&end, NULL);

    /* N = 4096
    TILE_SIZE, latency
    untiled, 4.31
    128, 4.02
    256, 4.03
    512, 3.97
    1024, 4.00
    */

    /* N = 8192
    TILE_SIZE, latency
    untiled, 34.28
    4, 40.59
    8, 37.42
    16, 35.75
    32, 35.43
    64, 34.52
    128, 33.82
    256, 33.68
    512, 33.65
    1024, 33.99
    2048, 34.39
    4096, 34.96
    */
    printf("time taken for tiled matmul (tiled on middle loop): %0.8lf\n", timeDiff(&start, &end));

    double checksum = 0.0;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            checksum += C[i][j];
        }
    }
    printf("sum of C: %0.8lf\n", checksum);
    return 0;
}