# Week 1 Specification: Monte Carlo Pi Estimator

## Goal

Implement a small, reproducible Monte Carlo estimator for pi.

## Public interface

`estimate_pi(n, seed)` returns a floating-point estimate of pi.

- `n` is the number of random points to sample and must be a positive integer.
- `seed` initializes a private pseudo-random number generator. Calls using the
  same `n` and `seed` must return the same result.
- Invalid values of `n` raise `ValueError`.

## Method

Sample `n` points uniformly from the unit square. Count the points satisfying
`x**2 + y**2 <= 1`, then return four times the fraction inside the quarter
circle.

## Acceptance check

With `n=1_000_000` and `seed=2026`, the estimate must be within `1e-2` of
`math.pi`.

