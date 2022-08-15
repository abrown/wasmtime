(; A minimal example of a parallel for--each thread of execution writes to the
same location in memory. This makes use of the implicit detail in the current
implementation that allows a CPU-run kernel to modify the module's memory.

Note how both the kernel and the check at the end of `_start` access offset 0 in
memory. This means they are overwriting the kernel code itself, which is a risk
in the "bag-of-bytes" paradigm. ;)

(module
    (import "wasi_ephemeral_parallel" "parallel_for" (func $for
        (param $kernel_start i32)
        (param $kernel_len i32)
        (param $num_iterations i32)
        (param $block_size i32)
        (param $in_buffers_start i32)
        (param $in_buffers_len i32)
        (param $out_buffers_start i32)
        (param $out_buffers_len i32)
        (result i32)))

    ;; The kernel here is the binary-encoded version of `parallel-for-kernel.wat`, using:
    ;; $ wat2wasm tests/wat/parallel-for-kernel.wat --enable-threads --output=- | xxd -g 1 -p | sed -r 's/.{2}/\\&/g'
    ;; The length is calculated using `wc -c` minus the final newline (and minus one again).
    (memory (export "memory") 1 1 shared)
    (data (i32.const 0) "\00\61\73\6d\01\00\00\00\01\07\01\60\03\7f\7f\7f\00\02\0d\01\00\06\6d\65\6d\6f\72\79\02\03\01\01\03\02\01\00\07\0a\01\06\6b\65\72\6e\65\6c\00\00\0a\0e\01\0c\00\20\00\41\01\6a\41\00\36\02\00\0b")

    (func (export "_start") (result i32)
        (call $for (i32.const 0) (i32.const 64) (i32.const 12) (i32.const 4) (i32.const 0) (i32.const 0) (i32.const 0) (i32.const 0))
        (; Check the parallel for returned 0 (success) and that the memory was updated by an invocation of the kernel--if so, return 0. ;)
        (i32.ne (i32.load (i32.const 0)) (i32.const 0))
        (i32.or))
)
