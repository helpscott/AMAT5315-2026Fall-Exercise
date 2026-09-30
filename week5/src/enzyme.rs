use anyhow::{ensure, Result};

unsafe extern "C" {
    fn enzyme_cube(x: f64, out: *mut f64);
    fn enzyme_step_primal(
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
    );
    fn enzyme_step_jvp(
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
    );
    fn enzyme_step_vjp(
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
    );
}

fn validate(n: usize, arrays: &[&[f64]]) -> Result<()> {
    for values in arrays {
        ensure!(values.len() == n, "kernel slice has length {}, expected {n}", values.len());
    }
    Ok(())
}

pub fn cube_check(x: f64) -> [f64; 3] {
    let mut out = [0.0; 3];
    unsafe { enzyme_cube(x, out.as_mut_ptr()) };
    out
}

pub fn primal(
    previous: &[f64],
    current: &[f64],
    velocity: &[f64],
    damping: &[f64],
    source: &[f64],
    nx: usize,
    nz: usize,
    dx: f64,
    dt: f64,
) -> Result<Vec<f64>> {
    let n = nx * nz;
    validate(n, &[previous, current, velocity, damping, source])?;
    let mut next = vec![0.0; n];
    unsafe {
        enzyme_step_primal(
            previous.as_ptr(), current.as_ptr(), velocity.as_ptr(), damping.as_ptr(),
            source.as_ptr(), next.as_mut_ptr(), nx, nz, dx, dt,
        )
    };
    Ok(next)
}

pub fn jvp(
    previous: &[f64],
    d_previous: &[f64],
    current: &[f64],
    d_current: &[f64],
    velocity: &[f64],
    d_velocity: &[f64],
    damping: &[f64],
    source: &[f64],
    nx: usize,
    nz: usize,
    dx: f64,
    dt: f64,
) -> Result<(Vec<f64>, Vec<f64>)> {
    let n = nx * nz;
    validate(n, &[previous, d_previous, current, d_current, velocity, d_velocity, damping, source])?;
    let mut next = vec![0.0; n];
    let mut d_next = vec![0.0; n];
    unsafe {
        enzyme_step_jvp(
            previous.as_ptr(), d_previous.as_ptr(), current.as_ptr(), d_current.as_ptr(),
            velocity.as_ptr(), d_velocity.as_ptr(), damping.as_ptr(), source.as_ptr(),
            next.as_mut_ptr(), d_next.as_mut_ptr(), nx, nz, dx, dt,
        )
    };
    Ok((next, d_next))
}

pub fn vjp(
    previous: &[f64],
    current: &[f64],
    velocity: &[f64],
    damping: &[f64],
    source: &[f64],
    output_adjoint: &[f64],
    nx: usize,
    nz: usize,
    dx: f64,
    dt: f64,
) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>)> {
    let n = nx * nz;
    validate(n, &[previous, current, velocity, damping, source, output_adjoint])?;
    let mut d_previous = vec![0.0; n];
    let mut d_current = vec![0.0; n];
    let mut d_velocity = vec![0.0; n];
    let mut next = vec![0.0; n];
    let mut d_next = output_adjoint.to_vec();
    unsafe {
        enzyme_step_vjp(
            previous.as_ptr(), d_previous.as_mut_ptr(), current.as_ptr(), d_current.as_mut_ptr(),
            velocity.as_ptr(), d_velocity.as_mut_ptr(), damping.as_ptr(), source.as_ptr(),
            next.as_mut_ptr(), d_next.as_mut_ptr(), nx, nz, dx, dt,
        )
    };
    Ok((d_previous, d_current, d_velocity))
}

