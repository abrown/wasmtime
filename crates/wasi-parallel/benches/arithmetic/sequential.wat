(; Add up 1 billion integers 4 consecutive times. ;)

(module
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

    (; Register the nstream kernel as reference-able ;)
    (table 1 funcref)
    (elem (i32.const 0) $add)

    (memory (export "memory") 1)

    (func (export "_start")
        (local $len i32)
        (local.set $len (i32.const 1000000000))

        (call $add (i32.const 0) (i32.const 8) (local.get $len))
        (call $add (i32.const 1) (i32.const 8) (local.get $len))
        (call $add (i32.const 2) (i32.const 8) (local.get $len))
        (call $add (i32.const 3) (i32.const 8) (local.get $len)))
)
