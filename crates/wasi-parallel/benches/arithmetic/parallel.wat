(; Add up 1 billion integers in parallel on 4 different cores. WARNING: no error
checking provided! ;)

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

    (; The parallel kernel. ;)
    (func $add (param $thread_id i32) (param $num_threads i32) (param $block_size i32)
        (local $i i32)
        (local $end i32)
        (local.set $i (i32.const 0))
        (local.set $end (i32.add (local.get $block_size) (local.get $thread_id)))

        (block
            (loop
                (local.set $i (i32.add (local.get $i) (i32.const 1)))
                (i32.ge_s (local.get $i) (local.get $end))
                (br_if 1)
                (br 0)
            )
        )

        (i32.store (i32.const 0) (local.get $i))
    )

    (; Register the kernel as reference-able. ;)
    (table (export "__indirect_function_table") 1 funcref)
    (elem (i32.const 0) $add)

    (memory (export "memory") 1)

    (func (export "_start")
        (local $len i32)
        (local $device i32)
        (local.set $len (i32.const 1000000000))

        (drop (call $get_device (i32.const 0x02) (i32.const 0x00)))
        (local.set $device (i32.load (i32.const 0x00)))

        (call $for
            (; function to run; SPIR-V section index ;)
            (i32.const 0)
            (; number of threads ;)
            (i32.const 4)
            (; block size ;)
            (local.get $len)
            (; list of input buffers, offset + length ;)
            (i32.const 0) (i32.const 0)
            (; list of output buffers, offset + length ;)
            (i32.const 0) (i32.const 0))

        drop)
)
