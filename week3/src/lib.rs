//! Two-dimensional square-lattice Ising model samplers.

#[derive(Clone, Debug)]
pub struct Rng {
    state: u64,
}
impl Rng {
    pub fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 { 0x9e3779b97f4a7c15 } else { seed },
        }
    }
    pub fn next_u64(&mut self) -> u64 {
        // xorshift64*: compact, deterministic, and fast enough for long runs.
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        x.wrapping_mul(0x2545f4914f6cdd1d)
    }
    pub fn unit(&mut self) -> f64 {
        (self.next_u64() as f64) / (u64::MAX as f64 + 1.0)
    }
    pub fn index(&mut self, n: usize) -> usize {
        (self.next_u64() as usize) % n
    }
}

#[derive(Clone, Debug)]
pub struct Ising {
    pub l: usize,
    pub spins: Vec<i8>,
    rng: Rng,
}
impl Ising {
    pub fn new(l: usize, seed: u64) -> Self {
        assert!(l >= 2);
        Self {
            l,
            spins: vec![1; l * l],
            rng: Rng::new(seed),
        }
    }
    #[inline]
    fn idx(&self, x: usize, y: usize) -> usize {
        y * self.l + x
    }
    #[inline]
    fn neighbors(&self, i: usize) -> [usize; 4] {
        let x = i % self.l;
        let y = i / self.l;
        [
            self.idx((x + 1) % self.l, y),
            self.idx((x + self.l - 1) % self.l, y),
            self.idx(x, (y + 1) % self.l),
            self.idx(x, (y + self.l - 1) % self.l),
        ]
    }
    #[inline]
    fn neighbor_sum(&self, i: usize) -> i8 {
        self.neighbors(i).iter().map(|&j| self.spins[j]).sum()
    }
    pub fn magnetization(&self) -> f64 {
        self.spins.iter().map(|&s| s as f64).sum::<f64>() / self.spins.len() as f64
    }
    pub fn energy(&self) -> f64 {
        let mut e = 0i32;
        for y in 0..self.l {
            for x in 0..self.l {
                let i = self.idx(x, y);
                e -= (self.spins[i] as i32) * (self.spins[self.idx((x + 1) % self.l, y)] as i32);
                e -= (self.spins[i] as i32) * (self.spins[self.idx(x, (y + 1) % self.l)] as i32);
            }
        }
        e as f64 / self.spins.len() as f64
    }
    /// One Metropolis sweep is L^2 independently sampled site proposals (with replacement).
    pub fn metropolis_sweep(&mut self, t: f64) -> usize {
        let mut accepted = 0;
        // The only possible energy changes are -8, -4, 0, 4, and 8.
        let table = [1.0, 1.0, 1.0, (-4.0 / t).exp(), (-8.0 / t).exp()];
        for _ in 0..self.spins.len() {
            let i = self.rng.index(self.spins.len());
            let de = 2 * (self.spins[i] as i32) * (self.neighbor_sum(i) as i32);
            let table_index = ((de + 8) / 4) as usize;
            if table[table_index] == 1.0 || self.rng.unit() < table[table_index] {
                self.spins[i] = -self.spins[i];
                accepted += 1;
            }
        }
        accepted
    }
    /// One Wolff cluster flip, returning the cluster size.
    pub fn wolff_move(&mut self, t: f64) -> usize {
        let p = 1.0 - (-2.0 / t).exp();
        let start = self.rng.index(self.spins.len());
        let old = self.spins[start];
        let mut marked = vec![false; self.spins.len()];
        let mut stack = vec![start];
        let mut cluster = Vec::new();
        marked[start] = true;
        while let Some(i) = stack.pop() {
            cluster.push(i);
            for j in self.neighbors(i) {
                if !marked[j] && self.spins[j] == old && self.rng.unit() < p {
                    marked[j] = true;
                    stack.push(j);
                }
            }
        }
        for &i in &cluster {
            self.spins[i] = -self.spins[i];
        }
        cluster.len()
    }
    pub fn state(&self) -> &[i8] {
        &self.spins
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn all_up_energy_is_minus_two() {
        let m = Ising::new(4, 1);
        assert!((m.energy() + 2.0).abs() < 1e-12);
    }
    #[test]
    fn flip_delta_matches_energy() {
        let mut m = Ising::new(4, 2);
        let before = m.energy();
        m.spins[0] = -1;
        let after = m.energy();
        assert!((after - before - 0.5).abs() < 1e-12);
    }
    #[test]
    fn same_seed_is_reproducible() {
        let mut a = Ising::new(8, 42);
        let mut b = Ising::new(8, 42);
        for _ in 0..4 {
            a.metropolis_sweep(2.3);
            b.metropolis_sweep(2.3);
        }
        assert_eq!(a.spins, b.spins);
    }
}
