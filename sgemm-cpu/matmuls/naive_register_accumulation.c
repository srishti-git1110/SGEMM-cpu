// ik i've given a terrible name to this file
#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#define N 4096

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

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float partial_sum = 0.0;
            for (int k = 0; k < N; k++) {
                partial_sum += A[i][k] * B[k][j];
            }
            C[i][j] = partial_sum;
        }
    }

    struct timeval end;
    gettimeofday(&end, NULL);

    // 199.866s
    printf("time taken for naive matmul with register optimization: %0.8lf\n", timeDiff(&start, &end));

    // to avoid dead code elimination
    double checksum = 0.0;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            checksum += C[i][j];
        }
    }
    printf("sum of C: %0.8lf\n", checksum);
    return 0;
}