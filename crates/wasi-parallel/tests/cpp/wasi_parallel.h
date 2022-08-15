
#ifndef WASM_PARALLEL_H_INCLUDED
#define WASM_PARALLEL_H_INCLUDED

#ifdef __cplusplus
extern "C" {
#endif

#define WASI_PARALLEL_IMPORT(func_name) __attribute__((import_module("wasi_ephemeral_parallel"), import_name(func_name)))

typedef int i32;

// C/C++ doesn't support a generic pointer for functions. Therefore,
//  we'll have to use an common extension that lets you cast function pointers
//  to data pointers.
typedef void* KernelFn;

typedef enum
{
    CPU = 0,
    DISCRETE_GPU,
    INTEGRATED_GPU,
} DeviceKind;

typedef enum
{
    Read = 0,
    Write,
    ReadWrite,
} BufferAccess;


WASI_PARALLEL_IMPORT("get_device") extern int get_device(DeviceKind hint, int* device);
WASI_PARALLEL_IMPORT("create_buffer") extern int create_buffer(int device, int size, BufferAccess access, int* buffer);
WASI_PARALLEL_IMPORT("parallel_for") extern int parallel_for(
    KernelFn worker_func,
    i32 num_threads, i32 block_size,
    const i32 *in_buffers_start, i32 in_buffers_len,
    const i32 *out_buffers_start, i32 out_buffers_len);

WASI_PARALLEL_IMPORT("read_buffer") extern int read_buffer(int buffer, void *data, int size);
WASI_PARALLEL_IMPORT("write_buffer") extern int write_buffer(void *data, int size, int buffer);

#ifdef __cplusplus
}
#endif

#endif // WASM_PARALLEL_H_INCLUDED
