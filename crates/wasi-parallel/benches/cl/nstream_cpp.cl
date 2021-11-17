
typedef float BufferArrayType;

__kernel void spir_main(__global BufferArrayType *context,
                        __global BufferArrayType *bufA,
                        __global BufferArrayType *bufB,
                        __global BufferArrayType *bufC) {
  int idx = get_global_id(0);
  const BufferArrayType scalar = context[1];

  bufA[idx] = bufA[idx] + bufB[idx] * scalar + bufC[idx];
}
