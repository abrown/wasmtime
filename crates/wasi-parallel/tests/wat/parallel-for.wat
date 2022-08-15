(; A minimal example of a parallel for--each thread of execution writes to the
same location in memory. This makes use of the implicit detail in the current
implementation that allows a CPU-run kernel to modify the module's memory. ;)

(module
    (import "wasi_ephemeral_parallel" "parallel_for" (func $for
        (param $kernel_index i32)
        (param $num_threads i32)
        (param $block_size i32)
        (param $in_buffers_start i32)
        (param $in_buffers_len i32)
        (param $out_buffers_start i32)
        (param $out_buffers_len i32)
        (result i32)))

    (table (export "__indirect_function_table") 1 funcref)
    (elem (i32.const 0) $kernel)

    (memory (export "memory") 1)
    (data (i32.const 0) "\00\00\00\00")

    (func $kernel (param $thread_id i32) (param $num_threads i32) (param $block_size i32)
        (i32.add (local.get $thread_id) (i32.const 1))
        (i32.store (i32.const 0)))

    (func (export "_start") (result i32)
        (call $for (i32.const 0) (i32.const 12) (i32.const 4) (i32.const 0) (i32.const 0) (i32.const 0) (i32.const 0))
        (; Check the parallel for returned 0 (success) and that the memory was updated by an invocation of the kernel--if so, return 0. ;)
        (i32.ne (i32.load (i32.const 0)) (i32.const 0))
        (i32.or))
)
