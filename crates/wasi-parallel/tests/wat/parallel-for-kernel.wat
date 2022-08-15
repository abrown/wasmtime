(module
    (memory (import "" "memory") 1 1 shared)
    (func $kernel (export "kernel") (param $thread_id i32) (param $num_iterations i32) (param $block_size i32) (param $in_buffers_array_start i32)
        (i32.add (local.get $thread_id) (i32.const 1))
        (i32.store (i32.const 0)))
)
