(; This hand-coded implementation of nstream runs the same nstream workload as `nstream-block.wat`
(see `verify.sh`) but sequentially, not in parallel. ;)

(module
    (; See, e.g., https://github.com/ParRes/Kernels/blob/default/Cxx11/nstream-tbb.cc#L132 ;)
    (func $nstream (param $thread_id i32) (param $num_threads i32) (param $block_size i32) (param $A i32) (param $A_len i32) (param $B i32) (param $B_len i32) (param $C i32) (param $C_len i32)
        (local $A_i i32)
        (local $B_i i32)
        (local $C_i i32)

        (block
            (loop
                (local.set $A_i (i32.add (local.get $A) (i32.mul (local.get $thread_id) (i32.const 4))))
                (local.set $B_i (i32.add (local.get $B) (i32.mul (local.get $thread_id) (i32.const 4))))
                (local.set $C_i (i32.add (local.get $C) (i32.mul (local.get $thread_id) (i32.const 4))))

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
                (local.set $thread_id (i32.add (local.get $thread_id) (i32.const 1)))
                (i32.ge_s (local.get $thread_id) (local.get $num_threads))
                (br_if 1)
                (br 0)
            )
        )
    )

    (memory (export "memory") 0x800)

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
        (local $len i32)
        (local $memA i32)
        (local $memB i32)
        (local $memC i32)

        (local.set $len (i32.const 0x2000000))
        (local.set $memA (local.get $len))
        (local.set $memB (i32.mul (local.get $len) (i32.const 2)))
        (local.set $memC (i32.mul (local.get $len) (i32.const 3)))

        (call $initialize (local.get $memA) (local.get $len) (f32.const 0))
        (call $initialize (local.get $memB) (local.get $len) (f32.const 2))
        (call $initialize (local.get $memC) (local.get $len) (f32.const 2))

        (call $nstream
            (; thread_id ;)
            (i32.const 0)
            (; number of threads; block size ;)
            (i32.div_u (local.get $len) (i32.const 4)) (i32.const 8)
            (; list of input buffers, offset + length ;)
            (local.get $memA) (local.get $len)
            (local.get $memB) (local.get $len)
            (local.get $memC) (local.get $len)))
)
