#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>

#define N 4096
#define TILE_K 32   // try 32, 64, 128

double timeDiff(struct timeval *start, struct timeval *end) {
    double start_sec = start->tv_sec + (start->tv_usec / 1000000.0);
    double end_sec = end->tv_sec + (end->tv_usec / 1000000.0);
    return end_sec - start_sec;
}

float A[N][N], B[N][N], C[N][N];

int main(int argc, char *argv[]) {
    // Same initialization as naive
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[i][j] = (float)(i + j) / (float) RAND_MAX;
            B[i][j] = (float)(i * j) / (float) RAND_MAX;
            C[i][j] = 0.0f;
        }
    }

    struct timeval start;
    gettimeofday(&start, NULL);

    // K-tiled matrix multiplication
    for (int i = 0; i < N; i++) {
        for (int kt = 0; kt < N; kt += TILE_K) {

            int kt_end = kt + TILE_K;
            if (kt_end > N) kt_end = N;

            for (int k = kt; k < kt_end; k++) {
                float a_ik = A[i][k];   // reuse A element

                for (int j = 0; j < N; j++) {
                    C[i][j] += a_ik * B[k][j];
                }
            }
        }
    }

    struct timeval end;
    gettimeofday(&end, NULL);

    printf("time taken for K-tiled matmul: %0.8lf\n",
           timeDiff(&start, &end));

    // Same checksum to avoid dead code elimination
    double checksum = 0.0;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            checksum += C[i][j];
        }
    }
    printf("sum of C: %0.8lf\n", checksum);

    return 0;
}
