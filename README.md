# SGEMM-cpu
Optimizing matmul on cpu!

## Setup
I am using [the following machine](https://www.apple.com/in/shop/buy-mac/macbook-pro/14-inch-space-black-standard-display-apple-m5-chip-with-10-core-cpu-and-10-core-gpu-16gb-memory-512gb) with a 10 core CPU (4 Performance cores+6 Efficiency cores). 

## Algorithmic complexity of Matrix Multiplication: Calculating the FLOPs required
Consider two matrices A (i x k) and B (k x j). The product of A and B, AB is a matrix C of shape (i x j).

For simplicity and without loss of generality, let's consider the matrices to be square so we have A (n x n), B (n x n) and their product C (n x n). One element (c1, c2) of C is defined as:

c_{c1,c2} = \sum_{x=1}^{n} a_{c1,x} b_{x,c2}

This is a total of 2n - 1 floating point operations (FLOPs) required to calculate one element of C -- n multiplication ops + (n-1) addition ops. A total of n2 elements need to be calculated in C and hence the total FLOPs:

(2n - 1) n2 = 2n3 - n2

As n grows bigger (asymptomatic, if you're feeling fancy), n2 becomes pretty negligible in comparison to 2n3 and hence can be ignored so the total FLOPs required is roughly 2n3. And the complexity is O(n3). 

For the purpose of this repo, I am keeping n=4096 which translates to roughly 137 GFLOPs.

## [Naive implementation](./sgemm-cpu/matmuls/naive.c)
When complied with the `-O3` flag which is the maximum level with safe optimizations, the latency is 299.289 sec. Full compilation command below:

All the further optimizations use the same flags to compile the code.







