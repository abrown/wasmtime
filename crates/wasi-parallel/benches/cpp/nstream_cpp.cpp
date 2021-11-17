
#include "../../tests/cpp/wasi_parallel.h"

#include <algorithm>
#include <cstdio>

void cpu_worker(int thread_id, int num_threads, int block_size,
                float *ctx, int ctx_len,
                float *A, int A_len,
                float *B, int B_len,
                float *C, int C_len)
{

    const int offset = thread_id * block_size;
    for (int b = offset; b < (offset + block_size); ++b)
    {
        A[b] += B[b] + ctx[1] * C[b];
    }
}

int main(int argc, char **argv)
{
    return 0;
}

const int buffer_size = 0x2000000;
int device;
int aBufH;
float aBuf[0x2000000];
int bBufH;
float bBuf[0x2000000];
int cBufH;
float cBuf[0x2000000];
int ctxBufH;
// opencl seems to have a problem with 0th element
float ctxBuf[2];
bool force_sequential;
int exec_mode;

// modes:
//  0. sequential
//  1. CPU
//  2. GPU
__attribute__((export_name("setup"))) void setup(int mode)
{
    force_sequential = mode == 0;
    exec_mode = mode;

    get_device(mode == 1 ? CPU: DISCRETE_GPU, &device);

    create_buffer(device, sizeof(aBuf), ReadWrite, &aBufH);
    create_buffer(device, sizeof(bBuf), Read, &bBufH);
    create_buffer(device, sizeof(cBuf), Read, &cBufH);
    create_buffer(device, sizeof(int) * 2, Read, &ctxBufH);
    std::fill_n(aBuf, buffer_size, 0);
    std::fill_n(bBuf, buffer_size, 2);
    std::fill_n(cBuf, buffer_size, 2);
    ctxBuf[0] = 0;
    ctxBuf[1] = 3;
    write_buffer(aBuf, sizeof(aBuf), aBufH);
    write_buffer(bBuf, sizeof(bBuf), bBufH);
    write_buffer(cBuf, sizeof(cBuf), cBufH);
    write_buffer(ctxBuf, sizeof(ctxBuf), ctxBufH);
}

__attribute__((export_name("run"))) void run()
{
    if (!force_sequential)
    {
        int in_buf[] = {ctxBufH, aBufH, bBufH, cBufH};
        const int num_threads = exec_mode == 1 ? 8 : 32;
        parallel_for(reinterpret_cast<void *>(cpu_worker),
            num_threads, buffer_size / num_threads,
            in_buf, sizeof(in_buf) / sizeof(int),
            0, 0);
    }
    else
    {
        cpu_worker(0, 1, buffer_size,
                   ctxBuf, 2,
                   aBuf, buffer_size,
                   bBuf, buffer_size,
                   cBuf, buffer_size);
    }
}
