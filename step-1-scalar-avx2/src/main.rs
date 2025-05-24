// GEMM with scalar and AVX 2 support for f64/f32
// Written by Rich Neale from mathsDOTearth

use std::arch::x86_64::*;
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

// x86_64 AVX2 f64
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

}
