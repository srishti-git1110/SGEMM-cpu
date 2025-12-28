# SGEMM-cpu
Optimizing matmul on cpu!

Also writing an accompanying [blog](https://srishti-git1110.github.io/blog/matmul-cpu/).

## Setup
I am using [the following machine](https://www.apple.com/in/shop/buy-mac/macbook-pro/14-inch-space-black-standard-display-apple-m5-chip-with-10-core-cpu-and-10-core-gpu-16gb-memory-512gb) with a 10 core CPU (4 Performance cores+6 Efficiency cores). 

## Algorithmic complexity of Matrix Multiplication: Calculating the FLOPs required
Consider two matrices $A (i \times k)$ and $B (k \times j)$. The product of $A$ and $B$, $AB$ is a matrix $C$ of shape $(i \times j)$.

For simplicity and without loss of generality, let's consider the matrices to be square so we have $A (n \times n)$, $B (n \times n)$ and their product $C (n \times n)$. One element $(c_1, c_2)$ of $C$ is defined as:

$c_{c1,c2} = \sum_{x=1}^{n} a_{c1,x} b_{x,c2}$

This is a total of 2n - 1 floating point operations (FLOPs) required to calculate one element of C -- $n$ multiplication ops + $(n-1)$ addition ops. A total of $n^2$ elements need to be calculated in $C$ and hence the total FLOPs:

$(2n - 1) n^2 = 2n^3 - n^2$

As $n$ grows bigger (asymptomatic, if you're feeling fancy), $n^2$ becomes pretty negligible in comparison to $2n^3$ and hence can be ignored so the total FLOPs required is roughly $2n^3$. And the complexity is $O(n^3)$. 

For the purpose of this repo, I am keeping $n=4096$ which translates to roughly 137 GFLOPs.

## [Naive implementation](./sgemm-cpu/matmuls/naive.c)
The matmul loop is this:

```C
for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < N; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
```

When complied with the `-O3` flag which is the maximum level with safe optimizations, the latency is **299.289 sec**. Full compilation command below:

```
gcc -Wall -O3 sgemm-cpu/matmuls/naive.c -o sgemm-cpu/matmuls/naive
```

All the further optimizations use the same flags to compile the code.

### [Local Accumulation](./sgemm-cpu/matmuls/naive_register_accumulation.c)

To make sure the compiler never issues a separate store and load instruction for the running partial sum and thus reduce some latency, we could accumulate the partial sum in a register variable, like so:


```C
for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float running_sum = 0.0;
            for (int k = 0; k < N; k++) {
                running_sum += A[i][k] * B[k][j];
            }
            C[i][j] = running_sum;
        }
    }
```
That brings the latency down to **199.866s**.

## [Loop reordering](./sgemm-cpu/matmuls/cache_aware.c)
Experimenting w/ different loop orders, the lowest latency of **4.49s** corresponds to order ikj down from 203.229s with order ijk which is a ~45x improvement already!

```C
for (int i = 0; i < N; i++) {
        for (int k = 0; k < N; k++) {
            for (int j = 0; j < N; j++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
```
