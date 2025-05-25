//! RISC-V RVV 1.0 support via a C function
#include <riscv_vector.h>
#include <stddef.h>

// Single‐precision float dot‐product via RVV 1.0
float vector_dot_f32(const float *x, const float *y, size_t n) {
    float dot = 0.0f;
    size_t i = 0;

    while (i < n) {
        // Set VL to process the remaining elements, using e32,m8 for
        // the widest vector we'll need
        size_t vl = __riscv_vsetvl_e32m8(n - i);

        // Load vectors of up to 8×32-bit floats
        vfloat32m8_t vx = __riscv_vle32_v_f32m8(&x[i], vl);
        vfloat32m8_t vy = __riscv_vle32_v_f32m8(&y[i], vl);

        // Element‐wise multiply
        vfloat32m8_t vprod = __riscv_vfmul_vv_f32m8(vx, vy, vl);

        // Prepare a scalar-vector of zeros to accumulate into
        vfloat32m1_t vzero = __riscv_vfmv_s_f_f32m1(0.0f, vl);

        // Ordered reduction: sum all lanes of vprod into one lane of vzero
        vfloat32m1_t vsum = __riscv_vfredosum_vs_f32m8_f32m1(vprod, vzero, vl);

        // Extract the scalar sum from lane 0
        float partial = __riscv_vfmv_f_s_f32m1_f32(vsum);

        dot += partial;
        i  += vl;
    }

    return dot;
}

// Double‐precision float dot‐product via RVV 1.0
double vector_dot_f64(const double *x, const double *y, size_t n) {
    double dot = 0.0;
    size_t i = 0;

    while (i < n) {
        // Use e64,m4 (4×64-bit lanes) for maximum throughput
        size_t vl = __riscv_vsetvl_e64m4(n - i);

        vfloat64m4_t vx = __riscv_vle64_v_f64m4(&x[i], vl);
        vfloat64m4_t vy = __riscv_vle64_v_f64m4(&y[i], vl);

        vfloat64m4_t vprod = __riscv_vfmul_vv_f64m4(vx, vy, vl);

        vfloat64m1_t vzero = __riscv_vfmv_s_f_f64m1(0.0, vl);

        vfloat64m1_t vsum = __riscv_vfredosum_vs_f64m4_f64m1(vprod, vzero, vl);

        double partial = __riscv_vfmv_f_s_f64m1_f64(vsum);

        dot += partial;
        i  += vl;
    }

    return dot;
}
