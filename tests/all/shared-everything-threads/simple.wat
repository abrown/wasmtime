(component
  (core module $m
    (import "thread" "spawn" (func $spawn (param (ref null func)) (param i32) (result i32)))
    (import "thread" "hw_concurrency" (func $hw_concurrency (result i32)))
    (func $start
      call $hw_concurrency
      i32.const 1
      i32.ne
      if
        unreachable
      end
    )
    (start $start)
  )

  (type $start (func (param "context" u32)))
  (core func $spawn (canon thread.spawn $start))
  (core func $concurrency (canon thread.hw_concurrency))
  (core instance (instantiate $m
    (with "thread" (instance
      (export "spawn" (func $spawn))
      (export "hw_concurrency" (func $concurrency))
    ))
  ))
)
