#include <stdio.h>
#include <sys/time.h>
#include <stdlib.h>
#include <arm_neon.h>

#define N 4096

#define TILE_I 128
#define TILE_J 128
#define TILE_K 512

#define IR 4
#define JR 4


double timeDiff(struct timeval *start, struct timeval *end) {
    double start_sec = start->tv_sec + (start->tv_usec / 1000000.0);
    double end_sec = end->tv_sec + (end->tv_usec / 1000000.0);
    return end_sec - start_sec;
}


float A[N][N], B[N][N], C[N][N];
float A_pack[TILE_I * TILE_K];
float B_pack[TILE_K * TILE_J];

int main(int argc, char *argv[]) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[i][j] = (float)(i + j) / (float)RAND_MAX;
            B[i][j] = (float)(i * j) / (float)RAND_MAX;
            C[i][j] = 0.0f;
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

                // pack A: 
                int a_pos = 0;
                for (int i = i_tile; i < iend; i += IR) {
                    for (int k = k_tile; k < kend; k++) {
                        A_pack[a_pos++] = A[i + 0][k];
                        A_pack[a_pos++] = A[i + 1][k];
                        A_pack[a_pos++] = A[i + 2][k];
                        A_pack[a_pos++] = A[i + 3][k];
                    }
                }

                // pack B
                int b_pos = 0;
                for (int j = j_tile; j < jend; j += JR) {
                    for (int k = k_tile; k < kend; k++) {
                        B_pack[b_pos++] = B[k][j + 0];
                        B_pack[b_pos++] = B[k][j + 1];
                        B_pack[b_pos++] = B[k][j + 2];
                        B_pack[b_pos++] = B[k][j + 3];
                    }
                }

                int a_micro_start = 0;
                for (int i = i_tile; i < iend; i += IR) {
                    int b_micro_start = 0;
                    for (int j = j_tile; j < jend; j += JR) {
                        float32x4_t c0 = vld1q_f32(&C[i+0][j]);
                        float32x4_t c1 = vld1q_f32(&C[i+1][j]);
                        float32x4_t c2 = vld1q_f32(&C[i+2][j]);
                        float32x4_t c3 = vld1q_f32(&C[i+3][j]);

                        float *a_ptr = &A_pack[a_micro_start];
                        float *b_ptr = &B_pack[b_micro_start];

                        for (int k = k_tile; k < kend; k++) {

                            float32x4_t a = vld1q_f32(a_ptr);
                            float32x4_t b = vld1q_f32(b_ptr);

                            c0 = vfmaq_n_f32(c0, b, vgetq_lane_f32(a, 0));
                            c1 = vfmaq_n_f32(c1,b, vgetq_lane_f32(a, 1));
                            c2 = vfmaq_n_f32(c2, b, vgetq_lane_f32(a, 2));
                            c3 = vfmaq_n_f32(c3, b, vgetq_lane_f32(a, 3));

                            a_ptr += IR;
                            b_ptr += JR;
                        }
                        vst1q_f32(&C[i+0][j], c0);
                        vst1q_f32(&C[i+1][j], c1);
                        vst1q_f32(&C[i+2][j], c2);
                        vst1q_f32(&C[i+3][j], c3);

                        b_micro_start += (kend - k_tile) * JR;
                    }
                    a_micro_start += (kend - k_tile) * IR;
                }
            }
        }
    }

    struct timeval end;
    gettimeofday(&end, NULL);
    printf("time taken for packed SIMD register-blocked matmul: %0.8lf\n", timeDiff(&start, &end));
    
    double checksum = 0.0;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            checksum += C[i][j];
        }
    }
    printf("sum of C: %0.8lf\n", checksum);
    return 0;
}