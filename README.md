# SGEMM-cpu
Optimizing matmul on cpu!

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

When complied with the `-O3` flag which is the maximum level with safe optimizations, the latency is 299.289 sec. Full compilation command below:

```
gcc -Wall -O3 sgemm-cpu/matmuls/naive.c -o sgemm-cpu/matmuls/naive
```

All the further optimizations use the same flags to compile the code.

## [Loop reordering](./sgemm-cpu/matmuls/cache_aware.c)
The naive implementation follows the most natural mathy way to calculate a matmul $C = AB$ -- element $C[0][0]$ is *fully* calculated first via a scalar product of the first row of $A$ with the first column of $B$. Element $C[0][1]$ is *fully* calculated next via the scalar product of the first row of $A$ with the second column of $B$, and so on. 

There are two key insights over here:
1. Languages like C store matrices in the memory in a row major format like this:

![row major order](figs/row-major.jpeg)

If we now follow the calculation of $C[0][0]$ in code, it's equivalent to completing the inner-most $k$ loop for $i=0, j=0$ and the following values from $A, B, C$ are accessed at each subsequent iteration of this loop:

![values accessed](figs/matmul-values-accessed.png)


Remember that data is fetched from the memory in the caches in the granularity of cache lines. One cache line on my machine is of size 128 bytes which is equivalent to 32 single precision values. 

So on the very first loop iteration $i=0, j=0, k=0$, when $A[0][0], B[0][0], C[0][0]$ are fetched from the memory, the cache looks something like:

![cache after first iter](figs/cache.png)


This is simply because a cache line loads contiguous values from the memory where the matrices are stored in a row major format.
On the second iteration $i=0, j=0, k=1$, we need $A[0][1], B[1][0], C[0][0]$ and while A[0][1] and C[0][0] are found in the cache, we have a miss for B[1][0] that we need to fetch from the memory. And this holds for each iteration of the loop. Easy to figure out why -- our cache line is only 128 bytes (32 floats) while  each subsequent loop iteration accesses a value from B that's 4096 values apart from the value accessed in the last iteration. And this high cache miss rate explains the high latency we saw w the naive implementation.


<!-- Hence, for the access pattern shown above, we're getting good cache hit rates for values of A and C but a very high miss rate for values of B -- at each loop iteration, we have a hit for $C[0][0]$, mostly hits for value of the first row of A but we have a cache miss for all the values of B involved ($B[0][0], B[1][0]...$). Simply because the cache line loads contigous values from the memory and owing to the row major storage in the memory, the different values of B involved in calculating $C[0][0]$ are 4096 values (4096x4 bytes) apart which well exceeds the size of a usual cache line.  -->

2. The second insight is not too difficult to understand -- if we just change the loop orders (eg. jik or jki etc. instead of the most natural ijk), we'll still get the correct matmul. It's also why I was italicizing *fully* above. Thing is that with different loop orders we're not fully calculating each element of C in one full iteration of the innermost loop but that doesn't hurt the correctness of the matmul and that's easy to realise. Alright. Given that, we note that some loop orders have a better overall cache hit rate as compared to the naive ijk order (btw some orders also have a worse rate than ijk!). And hence just by changing the orders, we'll be able to reduce the latency by not having to make as many high latency accesses to the memory. Experimenting w different orders, the lowest latency of 4.49s corresponds to order ikj down from 203.229s with order ijk which is a ~45x improvement already!

```C
for (int i = 0; i < N; i++) {
        for (int k = 0; k < N; k++) {
            for (int j = 0; j < N; j++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
```