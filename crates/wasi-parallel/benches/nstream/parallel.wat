(; This hand-coded implementation of nstream splits the work--a 32MB buffer (4 *
LLC)--evenly among 4 cores and uses wasi-parallel to distribute the work. See,
e.g., https://github.com/ParRes/Kernels/blob/default/Cxx11/nstream-tbb.cc#L132
for a higher-level implementation. ;)

(module
    (import "wasi_ephemeral_parallel" "get_device" (func $get_device
        (param $hint i32)
        (param $out_device i32)
        (result i32)))
    (import "wasi_ephemeral_parallel" "create_buffer" (func $create_buffer
        (param $device i32)
        (param $size i32)
        (param $access i32)
        (param $out_buffer i32)
        (result i32)))
    (import "wasi_ephemeral_parallel" "write_buffer" (func $write_buffer
        (param $data_offset i32)
        (param $data_len i32)
        (param $buffer i32)
        (result i32)))
    (import "wasi_ephemeral_parallel" "read_buffer" (func $read_buffer
        (param $buffer i32)
        (param $data_offset i32)
        (param $data_len i32)
        (result i32)))
    (import "wasi_ephemeral_parallel" "parallel_for" (func $for
        (param $worker_func_index i32)
        (param $num_threads i32)
        (param $block_size i32)
        (param $in_buffers_start i32)
        (param $in_buffers_len i32)
        (param $out_buffers_start i32)
        (param $out_buffers_len i32)
        (result i32)))

    (; The nstream kernel. ;)
    (func $nstream (param $thread_id i32) (param $num_threads i32) (param $block_size i32) (param $A i32) (param $A_len i32) (param $B i32) (param $B_len i32) (param $C i32) (param $C_len i32)
        (local $A_i i32)
        (local $B_i i32)
        (local $C_i i32)
        (local $i i32)
        (local $end i32)
        (local.set $i (i32.mul (i32.mul (local.get $thread_id) (local.get $block_size)) (i32.const 4)))
        (local.set $end (i32.add (local.get $i) (i32.mul (local.get $block_size) (i32.const 4))))

        (block
            (loop
                (local.set $A_i (i32.add (local.get $A) (local.get $i)))
                (local.set $B_i (i32.add (local.get $B) (local.get $i)))
                (local.set $C_i (i32.add (local.get $C) (local.get $i)))

                (; Offset to store: A[i] = ... ;)
                (local.get $A_i)
                (; Value to store A[i] + B[i] + 3.0 * C[i] ;)
                (f32.add
                    (f32.load (local.get $A_i))
                    (f32.add
                        (f32.load (local.get $B_i))
                        (f32.mul
                            (f32.const 3.0)
                            (f32.load (local.get $C_i)))))
                (f32.store)

                (; Loop control ;)
                (local.set $i (i32.add (local.get $i) (i32.const 4)))
                (i32.ge_s (local.get $i) (local.get $end))
                (br_if 1)
                (br 0)
            )
        )
    )

    (; Register the nstream kernel as reference-able ;)
    (table (export "__indirect_function_table") 1 funcref)
    (elem (i32.const 0) $nstream)

    (memory (export "memory") 0x800)

    (; Helper function to (inefficiently) initialize a block of memory. ;)
    (func $initialize (param $offset i32) (param $len i32) (param $value f32)
        (block
            (loop
                (local.set $len (i32.sub (local.get $len) (i32.const 4)))
                (f32.store (i32.add (local.get $offset) (local.get $len)) (local.get $value))
                (i32.le_s (local.get $len) (i32.const 0))
                (br_if 1)
                (br 0)
            )
        )
    )

    (func (export "_start")
        (local $return_area i32)
        (local $device i32)
        (local $len i32)
        (local $memA i32)
        (local $memB i32)
        (local $memC i32)
        (local $A i32)
        (local $B i32)
        (local $C i32)

        (local.set $return_area (i32.const 0x00))
        (local.set $len (i32.const 0x2000000))
        (local.set $memA (local.get $len))
        (local.set $memB (i32.mul (local.get $len) (i32.const 2)))
        (local.set $memC (i32.mul (local.get $len) (i32.const 3)))

        (; Set up the device. ;)
        (drop (call $get_device (i32.const 0x02) (local.get $return_area)))
        (local.set $device (i32.load (local.get $return_area)))

        (; Set up the buffers. ;)
        (drop (call $create_buffer (local.get $device) (local.get $len) (i32.const 0x01) (local.get $return_area)))
        (local.set $A (i32.load (local.get $return_area)))
        (drop (call $create_buffer (local.get $device) (local.get $len) (i32.const 0x00) (local.get $return_area)))
        (local.set $B (i32.load (local.get $return_area)))
        (drop (call $create_buffer (local.get $device) (local.get $len) (i32.const 0x00) (local.get $return_area)))
        (local.set $C (i32.load (local.get $return_area)))

        (call $initialize (local.get $memA) (local.get $len) (f32.const 0))
        (call $initialize (local.get $memB) (local.get $len) (f32.const 2))
        (call $initialize (local.get $memC) (local.get $len) (f32.const 2))

        (drop (call $write_buffer (local.get $memA) (local.get $len) (local.get $A)))
        (drop (call $write_buffer (local.get $memB) (local.get $len) (local.get $B)))
        (drop (call $write_buffer (local.get $memC) (local.get $len) (local.get $C)))

        (i32.store (i32.const 0) (local.get $A))
        (i32.store (i32.const 4) (local.get $B))
        (i32.store (i32.const 8) (local.get $C))

        (call $for
            (; function to run; SPIR-V section index ;)
            (i32.const 0)
            (; number of threads ;)
            (i32.const 8)
            (; block size ;)
            (i32.div_u (i32.div_u (local.get $len) (i32.const 4)) (i32.const 8))
            (; list of input buffers, offset + length ;)
            (i32.const 0) (i32.const 3)
            (; list of output buffers, offset + length ;)
            (i32.const 0) (i32.const 0))
        (drop)

        (; Clean up the memory. ;)
        (i32.store (i32.const 0) (i32.const 0))
        (i32.store (i32.const 4) (i32.const 0))
        (i32.store (i32.const 8) (i32.const 0)))
)
