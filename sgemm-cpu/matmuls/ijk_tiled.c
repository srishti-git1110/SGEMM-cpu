#include <stdio.h>
#include <sys/time.h>
#include <stdlib.h>

#define N 4096
#define TILE_I 128
#define TILE_J 128
#define TILE_K 512


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
    for (int i_tile = 0; i_tile < N; i_tile += TILE_I) {
        int iend = (i_tile + TILE_I < N) ? i_tile + TILE_I : N;

        for (int j_tile = 0; j_tile < N; j_tile += TILE_J) {
            int jend = (j_tile + TILE_J < N) ? j_tile + TILE_J : N;

            for (int k_tile = 0; k_tile < N; k_tile += TILE_K) {
                int kend = (k_tile + TILE_K < N) ? k_tile + TILE_K : N;

                for (int i = i_tile; i < iend; i++) {
                    for (int k = k_tile; k < kend; k++) {
                        float a_ik = A[i][k];
                        for (int j = j_tile; j < jend; j++) {
                            C[i][j] += a_ik * B[k][j];
                        }
                    }
                }
            }
        }
    }

    struct timeval end;
    gettimeofday(&end, NULL);

    /* N = 4096
    TILE_SIZE ikj, latency
    untiled, 4.31
    64 64 64, 4.15
    128 128 128, 3.47
    256 256 256, 5.00
    64 64 128, 3.32
    128 256 128, 3.16
    */

    /* N = 8192
    TILE_SIZE ikj, latency
    untiled, 34.28
    64 64 64, 34.91
    128 128 128, 26.208
    256 256 256, 41.07
    */
    printf("time taken for tiled matmul (tiled over ijk): %0.8lf\n", timeDiff(&start, &end));

    double checksum = 0.0;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            checksum += C[i][j];
        }
    }
    printf("sum of C: %0.8lf\n", checksum);
    return 0;
}