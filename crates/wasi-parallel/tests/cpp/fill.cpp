
#include "wasi_parallel.h"

void cpu_worker(int thread_id, int num_threads, int block_size, void *in_buffer, int *in_buffer_lens, void *out_buffers, int *out_buffers_lens)
{
    // GPU only tests
}

int main(int argc, char** argv)
{
    int dev = 0;
    get_device(DISCRETE_GPU, &dev);
    const int expected[] = {0, 1, 2, 3};
    const int element_count = sizeof(expected) / sizeof(expected[0]);
    int buffer = 0;
    create_buffer(dev, sizeof(expected), Write, &buffer);
    parallel_for((void*)cpu_worker,
                 1, element_count,
                 0, 0,
                 &buffer, 1);
    int data[element_count] = {};
    read_buffer(buffer, data, sizeof(data));
    for (int i = 0; i < element_count; ++i)
    {
        if (expected[i] != data[i])
        {
            return 1;
        }
    }

    return 0;
}
