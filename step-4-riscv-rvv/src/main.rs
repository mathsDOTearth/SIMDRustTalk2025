// GEMM with scalar, AVX 2, AVX512, NEON and RVV 1.0 support for f64/f32
// Written by Rich Neale from mathsDOTearth
//
// Uncomment the following two lins if using a version of
// rustc prior to 1.87.0 (2025-05-09)
// #![cfg_attr(target_arch = "x86_64", feature(avx512_target_feature))]
// #![cfg_attr(target_arch = "x86_64", feature(stdarch_x86_avx512))]
#![cfg_attr(target_arch = "riscv64", feature(asm, riscv_target_feature))]

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;
#[cfg(all(target_arch = "riscv64", target_feature = "v"))]
use core::arch::asm;
use rayon::prelude::*;
use rayon::current_num_threads;
use num_traits::{Zero, Num};
use unirand::MarsagliaUniRng;
use std::time::Instant;
use std::iter::Sum;

// SIMD abstraction
pub unsafe trait SimdElem: Copy + Sized {
    type Scalar: Copy + Send + Sync + std::ops::Add<Output = Self::Scalar> + std::ops::Mul<Output = Self::Scalar>;
    type Reg;
    const LANES: usize;
    unsafe fn zero() -> Self::Reg;
    unsafe fn load(ptr: *const Self::Scalar) -> Self::Reg;
    unsafe fn store(ptr: *mut Self::Scalar, v: Self::Reg);
    unsafe fn fmadd(acc: Self::Reg, a: Self::Reg, b: Self::Reg) -> Self::Reg;
    unsafe fn reduce(v: Self::Reg) -> Self::Scalar;
}
#[cfg(all(target_arch = "riscv64", target_feature = "v"))]
unsafe impl SimdElem for f64 {
    type Scalar = f64;
    /// We assume hardware VLEN=128 ⇒ 128/64 = 2 lanes
    type Reg    = ();
    const LANES : usize = 2;

    #[inline(always)]
    unsafe fn zero() -> Self::Reg {
        // clear v0
        unsafe { asm!("vmv.v.i v0, 0", options(nomem, nostack)); }
    }

    #[inline(always)]
    unsafe fn load(ptr: *const f64) -> Self::Reg {
        unsafe { 
        // set VL = 2
            asm!("vsetvli t0, {lanes}, e64,m1", lanes = const Self::LANES);
            // load contiguous 64-bit elements into v1
            asm!("vlse64.v v1, ({src}), t0", src = in(reg) ptr);
        }
    }

    #[inline(always)]
    unsafe fn store(ptr: *mut f64, _v: Self::Reg) {
        unsafe { 
        // set VL = 2
            asm!("vsetvl t0, {lanes}, e64,m1", lanes = const Self::LANES);
            // store v1 back to memory
            asm!("vse64.v v1, ({dst})", dst = in(reg) ptr);
        }
    }

    #[inline(always)]
    unsafe fn fmadd(_acc: Self::Reg, _a: Self::Reg, _b: Self::Reg) -> Self::Reg {
        // vfmadd.vv v1, v1, v2   ; v1 += v1 * v2
        unsafe { asm!("vfmadd.vv v1, v1, v2", options(nomem, nostack)); }
    }

    #[inline(always)]
    unsafe fn reduce(_v: Self::Reg) -> Self::Scalar {
        unsafe { 
            // spill v1 to stack
            let mut buf = [0f64; Self::LANES];
            asm!("vsetvli t0, {lanes}, e64,m1", lanes = const Self::LANES);
            asm!("vse64.v v1, ({dst})", dst = in(reg) buf.as_mut_ptr());
            buf.iter().copied().sum()
        }
    }
}
// See RISC-V Vector intrinsics spec for vsetvl, vlse64, vfmadd, vse64 :contentReference[oaicite:0]{index=0}

// ------------------------------------------------------------------
// RISC-V RVV 1.0 support (VLEN=128) for f32
// ------------------------------------------------------------------

#[cfg(all(target_arch = "riscv64", target_feature = "v"))]
unsafe impl SimdElem for f32 {
    type Scalar = f32;
    /// 128/32 = 4 lanes
    type Reg    = ();
    const LANES : usize = 4;

    #[inline(always)]
    unsafe fn zero() -> Self::Reg {
        unsafe { asm!("vmv.v.i v0, 0", options(nomem, nostack)); }
    }

    #[inline(always)]
    unsafe fn load(ptr: *const f32) -> Self::Reg {
        unsafe { 
            asm!("vsetvli t0, {lanes}, e32,m1", lanes = const Self::LANES);
            asm!("vlse32.v v1, ({src}), t0", src = in(reg) ptr);
        }
    }

    #[inline(always)]
    unsafe fn store(ptr: *mut f32, _v: Self::Reg) {
        unsafe { 
            asm!("vsetvl t0, {lanes}, e32,m1", lanes = const Self::LANES);
            asm!("vse32.v v1, ({dst})", dst = in(reg) ptr);
        }
    }

    #[inline(always)]
    unsafe fn fmadd(_acc: Self::Reg, _a: Self::Reg, _b: Self::Reg) -> Self::Reg {
        unsafe { asm!("vfmadd.vv v1, v1, v2", options(nomem, nostack)); }
    }

    #[inline(always)]
    unsafe fn reduce(_v: Self::Reg) -> Self::Scalar {
        unsafe { 
            let mut buf = [0f32; Self::LANES];
            asm!("vsetvli t0, {lanes}, e32,m1", lanes = const Self::LANES);
            asm!("vse32.v v1, ({dst})", dst = in(reg) buf.as_mut_ptr());
            buf.iter().copied().sum()
        }
    }
}

// x86_64 AVX2 f64
#[cfg(all(target_arch = "x86_64", not(target_feature = "avx512f")))]
unsafe impl SimdElem for f64 {
    type Scalar = f64;
    type Reg = __m256d;
    const LANES: usize = 4;

    #[inline(always)]
    unsafe fn zero() -> __m256d {
        unsafe { _mm256_setzero_pd() }
    }

    #[inline(always)]
    unsafe fn load(ptr: *const f64) -> __m256d {
        unsafe { _mm256_loadu_pd(ptr) }
    }

    #[inline(always)]
    unsafe fn store(ptr: *mut f64, v: __m256d) {
        unsafe { _mm256_storeu_pd(ptr, v) }
    }

    #[inline(always)]
    unsafe fn fmadd(acc: __m256d, a: __m256d, b: __m256d) -> __m256d {
        unsafe { _mm256_fmadd_pd(a, b, acc) }
    }

    #[inline(always)]
    unsafe fn reduce(v: __m256d) -> f64 {
        unsafe {
            let hi = _mm256_extractf128_pd(v, 1);
            let mut lo = _mm256_castpd256_pd128(v);
            lo = _mm_add_pd(lo, hi); 
            lo = _mm_hadd_pd(lo, lo); 
            _mm_cvtsd_f64(lo)
        }
    }
}

// x86_64 AVX2 f64
#[cfg(all(target_arch = "x86_64", not(target_feature = "avx512f")))]
unsafe impl SimdElem for f32 {
    type Scalar = f32;
    type Reg    = __m256;
    const LANES : usize = 8;
    #[inline(always)]
    unsafe fn zero() -> __m256 {
        unsafe { _mm256_setzero_ps() }
    }

    #[inline(always)]
    unsafe fn load(ptr: *const f32) -> __m256 {
        unsafe { _mm256_loadu_ps(ptr) }
    }

    #[inline(always)]
    unsafe fn store(ptr: *mut f32, v: __m256) {
        unsafe { _mm256_storeu_ps(ptr, v) }
    }

    #[inline(always)]
    unsafe fn fmadd(acc: __m256, a: __m256, b: __m256) -> __m256 {
        unsafe { _mm256_fmadd_ps(a, b, acc) }
    }

    #[inline(always)]
    unsafe fn reduce(v: __m256) -> f32 {
        unsafe {
            // 1. add high 128 to low 128
            let hi  = _mm256_extractf128_ps(v, 1);
            let mut lo = _mm256_castps256_ps128(v);
            lo = _mm_add_ps(lo, hi);               // four partial sums
        
            // 2. horizontal add inside the 128-bit lane
            let lo = _mm_hadd_ps(lo, lo);          // [a0+a1, a2+a3, \u2026]
            let lo = _mm_hadd_ps(lo, lo);          // [total,  ?,  ?,  ?]
        
            _mm_cvtss_f32(lo)                      // pick the scalar
        }
    } 
}

// x86_64 AVX512 f64
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
unsafe impl SimdElem for f64 {
    type Scalar = f64;
    type Reg    = __m512d;
    const LANES : usize = 8;

    #[inline(always)]
    unsafe fn zero() -> __m512d { 
        unsafe {_mm512_setzero_pd() }
    }

    #[inline(always)]
    unsafe fn load(ptr: *const f64) -> __m512d {
        unsafe { _mm512_loadu_pd(ptr) }
    }

    #[inline(always)] 
    unsafe fn store(ptr: *mut f64, v: __m512d) {
        unsafe { _mm512_storeu_pd(ptr, v) }
    }

    #[inline(always)] 
    unsafe fn fmadd(acc: __m512d, a: __m512d, b: __m512d) -> __m512d {
        unsafe { _mm512_fmadd_pd(a, b, acc) }
    }

    #[inline(always)] 
    unsafe fn reduce(v: __m512d) -> f64 {
        let mut buf = [0f64; 8];
        unsafe { _mm512_storeu_pd(buf.as_mut_ptr(), v) };
        buf.iter().copied().sum()
    }
}

// x86_64 AVX512 f32
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
unsafe impl SimdElem for f32 {
    type Scalar = f32;
    type Reg    = __m512;
    const LANES : usize = 16;

    #[inline(always)] 
    unsafe fn zero() -> __m512 { 
        unsafe { _mm512_setzero_ps() }
    }

    #[inline(always)] 
    unsafe fn load(ptr: *const f32) -> __m512 { 
        unsafe { _mm512_loadu_ps(ptr) }
    }

    #[inline(always)] 
    unsafe fn store(ptr: *mut f32, v: __m512) { 
        unsafe { _mm512_storeu_ps(ptr, v) }
    }

    #[inline(always)] unsafe fn fmadd(acc: __m512, a: __m512, b: __m512) -> __m512 {
        unsafe { _mm512_fmadd_ps(a, b, acc) }
    }

    #[inline(always)] unsafe fn reduce(v: __m512) -> f32 {
        let mut buf = [0f32; 16];
        unsafe { _mm512_storeu_ps(buf.as_mut_ptr(), v) };
        buf.iter().copied().sum()
    }
}

// aarch64 NEON f64
#[cfg(target_arch = "aarch64")]
unsafe impl SimdElem for f64 {
    type Scalar = f64;
    type Reg    = float64x2_t;
    const LANES : usize = 2;

    #[inline(always)]
    unsafe fn zero() -> float64x2_t { 
        unsafe {vdupq_n_f64(0.0) }
    }

    #[inline(always)] 
    unsafe fn load(ptr: *const f64) -> float64x2_t { 
        unsafe { vld1q_f64(ptr) }
    }

    #[inline(always)]
    unsafe fn store(ptr: *mut f64, v: float64x2_t) { 
        unsafe { vst1q_f64(ptr, v) }
    }

    #[inline(always)] unsafe fn fmadd(acc: float64x2_t, a: float64x2_t, b: float64x2_t) -> float64x2_t {
        unsafe { vfmaq_f64(acc, a, b) }
    }

    #[inline(always)] unsafe fn reduce(v: float64x2_t) -> f64 {
        let arr: [f64;2] = unsafe { std::mem::transmute(v) }; 
        arr[0] + arr[1]
    }
}

// aarch64 NEON f32
#[cfg(target_arch = "aarch64")]
unsafe impl SimdElem for f32 {
    type Scalar = f32;
    type Reg    = float32x4_t;
    const LANES : usize = 4;

    #[inline(always)]
    unsafe fn zero() -> float32x4_t { 
        unsafe { vdupq_n_f32(0.0) }
    }

    #[inline(always)] unsafe fn load(ptr: *const f32) -> float32x4_t { 
        unsafe { vld1q_f32(ptr) }
    }

    #[inline(always)] unsafe fn store(ptr: *mut f32, v: float32x4_t) { 
        unsafe { vst1q_f32(ptr, v) }
    }

    #[inline(always)] unsafe fn fmadd(acc: float32x4_t, a: float32x4_t, b: float32x4_t) -> float32x4_t {
        unsafe { vfmaq_f32(acc, a, b) }
    }
    #[inline(always)] unsafe fn reduce(v: float32x4_t) -> f32 {
        let arr: [f32;4] = unsafe { std::mem::transmute(v) };
        arr.iter().copied().sum()
    }
}

pub unsafe fn dot_generic<E: SimdElem>(a: &[E::Scalar], b: &[E::Scalar]) -> E::Scalar {
    let mut i = 0; 
    let len = a.len(); 
    let mut acc = unsafe { E::zero() };
    while i + E::LANES <= len {
        let va = unsafe { E::load(a.as_ptr().add(i)) };
        let vb = unsafe { E::load(b.as_ptr().add(i)) };
        acc = unsafe { E::fmadd(acc, va, vb) };
        i += E::LANES;
    }
    let mut total = unsafe { E::reduce(acc) };
    while i < len { total = total + (a[i] * b[i]); i += 1; }
    total
}


// Transpose Array
pub fn transpose<T: Copy + Default>(m: &[Vec<T>]) -> Vec<Vec<T>> {
    let r = m.len(); let c = m[0].len(); let mut t = vec![vec![T::default(); r]; c];
    for i in 0..r { for j in 0..c { t[j][i] = m[i][j]; }}
    t
}

// Scalar GEMM f64/f32
pub fn gemm_scalar_f64(alpha: f64, a: &[Vec<f64>], b_t: &[Vec<f64>], beta: f64, c: &mut [Vec<f64>], parallel: bool) {
    let rows = a.len(); let cols = b_t.len();
    if parallel {
        c.par_iter_mut().enumerate().for_each(|(i, row_c)| {
            for j in 0..cols { 
                let mut sum=0f64; 
                for k in 0..a[0].len() { 
                    sum+=a[i][k]*b_t[j][k]; 
                } 
                row_c[j]=alpha*sum+beta*row_c[j]; 
            }
        });
    } else {
        for i in 0..rows { 
            for j in 0..cols { 
                let mut sum=0f64; 
                for k in 0..a[0].len() { 
                    sum+=a[i][k]*b_t[j][k]; 
                } 
                c[i][j]=alpha*sum+beta*c[i][j]; 
            }
        }
    }
}
pub fn gemm_scalar_f32(alpha: f32, a: &[Vec<f32>], b_t: &[Vec<f32>], beta: f32, c: &mut [Vec<f32>], parallel: bool) {
    let rows = a.len();
    let cols = b_t.len();
    if parallel {
        c.par_iter_mut().enumerate().for_each(|(i, row_c)| {
            for j in 0..cols {
                let mut sum = 0f32;
                for k in 0..a[0].len() {
                    sum += a[i][k] * b_t[j][k];
                }
                row_c[j] = alpha * sum + beta * row_c[j];
            }
        });
    } else {
        for i in 0..rows {
            for j in 0..cols {
                let mut sum = 0f32;
                for k in 0..a[0].len() {
                    sum += a[i][k] * b_t[j][k];
                }
                c[i][j] = alpha * sum + beta * c[i][j];
            }
        }
    }
}

// Convenience wrappers for SIMD
pub fn gemm_f64(alpha: f64, a: &[Vec<f64>], b_t: &[Vec<f64>], beta: f64, c: &mut [Vec< f64>], parallel: bool) {
    if parallel {
        c.par_iter_mut().enumerate().for_each(|(i, row_c)| {
            for j in 0..b_t.len() {
                let sum = unsafe { dot_generic::<f64>(&a[i], &b_t[j]) };
                row_c[j] = alpha * sum + beta * row_c[j];
            }
        });
    } else {
        for i in 0..a.len() {
            for j in 0..b_t.len() {
                let sum = unsafe { dot_generic::<f64>(&a[i], &b_t[j]) };
                c[i][j] = alpha * sum + beta * c[i][j];
            }
        }
    }
}

pub fn gemm_f32(alpha: f32, a: &[Vec<f32>], b_t: &[Vec<f32>], beta: f32, c: &mut [Vec<f32>], parallel: bool) {
    if parallel {
        c.par_iter_mut().enumerate().for_each(|(i, row_c)| {
            for j in 0..b_t.len() {
                let sum = unsafe { dot_generic::<f32>(&a[i], &b_t[j]) };
                row_c[j] = alpha * sum + beta * row_c[j];
            }
        });
    } else {
        for i in 0..a.len() {
            for j in 0..b_t.len() {
                let sum = unsafe { dot_generic::<f32>(&a[i], &b_t[j]) };
                c[i][j] = alpha * sum + beta * c[i][j];
            }
        }
    }
}

// Scalar DOT implementation
pub fn dot_scalar<T: SimdElem>(a: &[T::Scalar], b: &[T::Scalar]) -> T::Scalar
where
    T::Scalar: Copy + Num + Zero + Sum,
{
    a.iter().zip(b.iter()).map(|(x, y)| *x * *y).sum()
}

// Parallel Scalar DOT implementation
pub fn dot_scalar_parallel<T: SimdElem>(a: &[T::Scalar], b: &[T::Scalar]) -> T::Scalar
where
    T::Scalar: Copy + Send + Sync + Num + Zero + Sum,
{
    a.par_iter().zip(b.par_iter()).map(|(x, y)| *x * *y).sum()
}

// SIMD DOT with Rayon parallel implementation
pub fn dot_simd_parallel<T: SimdElem>(a: &[T::Scalar], b: &[T::Scalar]) -> T::Scalar
where
    T::Scalar: Copy + Send + Sync + Num + Zero + Sum,
{
    let chunks = a.len() / T::LANES;
    let simd_sum: T::Scalar = (0..chunks)
        .into_par_iter()
        .map(|chunk| unsafe {
            let offset = chunk * T::LANES;
            let va = T::load(a.as_ptr().add(offset));
            let vb = T::load(b.as_ptr().add(offset));
            T::reduce(T::fmadd(T::zero(), va, vb))
        })
        .sum();

    let remainder: T::Scalar = a[chunks * T::LANES..]
        .iter()
        .zip(b[chunks * T::LANES..].iter())
        .map(|(x, y)| *x * *y)
        .sum();

    simd_sum + remainder
}

pub fn dot_simd_parallel_chunks<T: SimdElem>(a: &[T::Scalar], b: &[T::Scalar]) -> T::Scalar
where
    T::Scalar: Copy + Send + Sync + Num + Zero + Sum,
{
    const MIN_CHUNK_SIZE: usize = 1 << 14;
    a.par_chunks(MIN_CHUNK_SIZE)
        .zip(b.par_chunks(MIN_CHUNK_SIZE))
        .map(|(chunk_a, chunk_b)| unsafe { dot_generic::<T>(chunk_a, chunk_b) })
        .sum()
}

// check tollerance of results
// Matrix element trait for comparison
pub trait MatrixElem: Copy + PartialOrd + std::ops::Sub<Output=Self> {
    fn abs(self) -> Self;
    fn tol() -> Self;
    fn label() -> &'static str;
}
impl MatrixElem for f64 {
    fn abs(self) -> Self { f64::abs(self) }
    fn tol() -> Self { 1e-8 }
    fn label() -> &'static str { "f64" }
}
impl MatrixElem for f32 {
    fn abs(self) -> Self { f32::abs(self) }
    fn tol() -> Self { 1e-3 }
    fn label() -> &'static str { "f32" }
}

// Generic correctness checker
pub fn check_matrix<T: MatrixElem + std::fmt::Display>(c1: &[Vec<T>], c2: &[Vec<T>], n: usize) {
    let tol = T::tol();
    for i in 0..n {
        for j in 0..n {
            let diff = (c1[i][j] - c2[i][j]).abs();
            if diff > tol {
                panic!("{} mismatch at ({},{}): ref={} vs simd={}",
                       T::label(), i, j, c1[i][j], c2[i][j]);
            }
        }
    }
    println!("✔ {} SIMD‑parallel matches reference (tol={})", T::label(), tol);
}
// ----------------------
// main with full benchmarking
// ----------------------
fn main() {
    const N: usize = 1024;
    let mut rng = MarsagliaUniRng::new(); rng.rinit(170);

    // f32 xGEMM data ench mark test run
    let a32: Vec<Vec<f32>> = (0..N).map(|_| (0..N).map(|_| rng.uni() as f32).collect()).collect();
    let bt32 = transpose(&a32);
    let mut c32 = vec![vec![0f32; N]; N];
    let mut c32_scalar = vec![vec![0f32; N]; N];

    println!("Benchmarking f32 xGEMM");
    let t4 = Instant::now(); gemm_scalar_f32(1.0, &a32, &bt32, 0.0, &mut c32_scalar, false); let f32_scalar = t4.elapsed();
    println!("f32 scalar: {:.2?}", f32_scalar);

    let t5 = Instant::now(); gemm_scalar_f32(1.0, &a32, &bt32, 0.0, &mut c32, true); let f32_par = t5.elapsed();
    println!("f32 parallel: {:.2?}", f32_par);
    println!("Speedup: {:.2}", f32_scalar.as_secs_f64() / f32_par.as_secs_f64());
    check_matrix(&c32, &c32_scalar, N);

    let t6 = Instant::now(); gemm_f32(1.0, &a32, &bt32, 0.0, &mut c32, false); let f32_vec = t6.elapsed();
    println!("f32 vector: {:.2?}", f32_vec);
    println!("Speedup: {:.2}", f32_scalar.as_secs_f64() / f32_vec.as_secs_f64());
    check_matrix(&c32, &c32_scalar, N);

    let t7 = Instant::now(); gemm_f32(1.0, &a32, &bt32, 0.0, &mut c32, true); let f32_vec_par = t7.elapsed();
    println!("f32 vector parallel: {:.2?}", f32_vec_par);
    println!("Speedup: {:.2}", f32_scalar.as_secs_f64() / f32_vec_par.as_secs_f64());
    check_matrix(&c32, &c32_scalar, N);

    println!(" ");

    // f64 xGEMM data bench mark test run
    let a64: Vec<Vec<f64>> = (0..N).map(|_| (0..N).map(|_| rng.uni() as f64).collect()).collect();
    let bt64 = transpose(&a64);
    let mut c64 = vec![vec![0f64; N]; N];
    let mut c64_scalar = vec![vec![0f64; N]; N];

    println!("Benchmarking f64 xGEMM");
    let t0 = Instant::now(); gemm_scalar_f64(1.0, &a64, &bt64, 0.0, &mut c64_scalar, false); let f64_scalar = t0.elapsed();
    println!("f64 scalar: {:.2?}", f64_scalar);

    let t1 = Instant::now(); gemm_scalar_f64(1.0, &a64, &bt64, 0.0, &mut c64, true); let f64_par = t1.elapsed();
    println!("f64 parallel: {:.2?}", f64_par);
    println!("Speedup: {:.2}", f64_scalar.as_secs_f64() / f64_par.as_secs_f64());
    check_matrix(&c64, &c64_scalar, N);

    let t2 = Instant::now(); gemm_f64(1.0, &a64, &bt64, 0.0, &mut c64, false); let f64_vec = t2.elapsed();
    println!("f64 vector: {:.2?}", f64_vec);
    println!("Speedup: {:.2}", f64_scalar.as_secs_f64() / f64_vec.as_secs_f64());
    check_matrix(&c64, &c64_scalar, N);
    
    let t3 = Instant::now(); gemm_f64(1.0, &a64, &bt64, 0.0, &mut c64, true); let f64_vec_par = t3.elapsed();
    println!("f64 vector parallel: {:.2?}", f64_vec_par);
    println!("Speedup: {:.2}", f64_scalar.as_secs_f64() / f64_vec_par.as_secs_f64());
    check_matrix(&c64, &c64_scalar, N);

    println!(" ");
    println!("Number of Rayon Threads: {}" , current_num_threads());

    #[cfg(target_arch = "aarch64")]
    {
        println!("Running With ARM NEON");
    }
        #[cfg(all(target_arch = "x86_64", not(target_feature = "avx512f")))]
    {
        println!("Running With x86_64 AVX2");
    }
    #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
    {
        println!("Running With x86_64 AVX512");
    }
}
