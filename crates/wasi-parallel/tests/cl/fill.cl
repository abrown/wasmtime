
__kernel void spir_main(__global int *buffer) {
    int id = get_global_id(0);
    buffer[id] = id;
}
