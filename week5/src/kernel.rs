#![no_std]
#![feature(autodiff)]

use core::autodiff::{autodiff_forward, autodiff_reverse};

#[autodiff_forward(step_forward, Dual, Dual, Dual, Const, Const, Dual, Const, Const, Const, Const)]
#[autodiff_reverse(step_reverse, Duplicated, Duplicated, Duplicated, Const, Const, Duplicated, Const, Const, Const, Const)]
unsafe fn step(
    previous: *const f64,
    current: *const f64,
    velocity: *const f64,
    damping: *const f64,
    source: *const f64,
    next: *mut f64,
    nx: usize,
    nz: usize,
    dx: f64,
    dt: f64,
) {
    let n = nx * nz;
    let mut k = 0;
    while k < n {
        let z = k / nx;
        let x = k - z * nx;
        if x == 0 || z == 0 || x + 1 == nx || z + 1 == nz {
            unsafe { *next.add(k) = 0.0 };
        } else {
            let center = unsafe { *current.add(k) };
            let lap = (unsafe { *current.add(k - 1) }
                + unsafe { *current.add(k + 1) }
                + unsafe { *current.add(k - nx) }
                + unsafe { *current.add(k + nx) }
                - 4.0 * center)
                / (dx * dx);
            let sigma = unsafe { *damping.add(k) };
            let c = unsafe { *velocity.add(k) };
            let numerator = 2.0 * center
                - (1.0 - sigma * dt) * unsafe { *previous.add(k) }
                + dt * dt * (c * c * lap + unsafe { *source.add(k) });
            unsafe { *next.add(k) = numerator / (1.0 + sigma * dt) };
        }
        k += 1;
    }
}

#[autodiff_forward(cube_forward, Dual, Dual)]
#[autodiff_reverse(cube_reverse, Active, Active)]
fn cube(x: f64) -> f64 {
    x * x * x
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn enzyme_cube(x: f64, out: *mut f64) {
    let (y, dy) = cube_forward(x, 1.0);
    let (_, adjoint) = cube_reverse(x, 1.0);
    unsafe {
        *out = y;
        *out.add(1) = dy;
        *out.add(2) = adjoint;
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn enzyme_step_primal(
    previous: *const f64,
    current: *const f64,
    velocity: *const f64,
    damping: *const f64,
    source: *const f64,
    next: *mut f64,
    nx: usize,
    nz: usize,
    dx: f64,
    dt: f64,
) {
    unsafe { step(previous, current, velocity, damping, source, next, nx, nz, dx, dt) }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn enzyme_step_jvp(
    previous: *const f64,
    d_previous: *const f64,
    current: *const f64,
    d_current: *const f64,
    velocity: *const f64,
    d_velocity: *const f64,
    damping: *const f64,
    source: *const f64,
    next: *mut f64,
    d_next: *mut f64,
    nx: usize,
    nz: usize,
    dx: f64,
    dt: f64,
) {
    unsafe {
        step_forward(
            previous,
            d_previous,
            current,
            d_current,
            velocity,
            d_velocity,
            damping,
            source,
            next,
            d_next,
            nx,
            nz,
            dx,
            dt,
        )
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn enzyme_step_vjp(
    previous: *const f64,
    d_previous: *mut f64,
    current: *const f64,
    d_current: *mut f64,
    velocity: *const f64,
    d_velocity: *mut f64,
    damping: *const f64,
    source: *const f64,
    next: *mut f64,
    d_next: *mut f64,
    nx: usize,
    nz: usize,
    dx: f64,
    dt: f64,
) {
    unsafe {
        step_reverse(
            previous,
            d_previous,
            current,
            d_current,
            velocity,
            d_velocity,
            damping,
            source,
            next,
            d_next,
            nx,
            nz,
            dx,
            dt,
        )
    }
}

unsafe extern "C" {
    fn abort() -> !;
}

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    unsafe { abort() }
}
