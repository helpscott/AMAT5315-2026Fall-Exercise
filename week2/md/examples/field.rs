use md::pair::{energy, force};
use std::io::{self, BufWriter, Write};

fn main() -> io::Result<()> {
    let mut output = BufWriter::new(io::stdout().lock());
    writeln!(output, "x,y,energy,fx,fy")?;
    let n = 241;
    let extent = 2.4;

    for iy in 0..n {
        let y = -extent + 2.0 * extent * iy as f64 / (n - 1) as f64;
        for ix in 0..n {
            let x = -extent + 2.0 * extent * ix as f64 / (n - 1) as f64;
            let r = x.hypot(y);
            // The potential is singular at the fixed atom itself.
            let (potential, fx, fy) = if r == 0.0 {
                (f64::NAN, 0.0, 0.0)
            } else {
                let radial_force = force(r);
                (energy(r), radial_force * x / r, radial_force * y / r)
            };
            writeln!(output, "{x:.12e},{y:.12e},{potential:.12e},{fx:.12e},{fy:.12e}")?;
        }
    }
    Ok(())
}
