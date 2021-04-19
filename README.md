# Kiley -- Consensus module

Author: Bansho Masutani

Email: ban-m@g.ecc.u-tokyo.ac.jp


## Overview

In this crate, 

- (Banded-)Ternary alignment with edit distance ,
- (Banded-)Bialignment with edit distance,
- (Banded-)pair hidden Markov model with forward/backward/viterbi algorithm, and
- (Banded-)generalized pair hidden Markov model with forward/backward/viterbi/pseudo-Baum-Whelch algorithm 

are implemented. In addition, there is a function to estimate the maximum-likelihood sequence based on a pair hidden Markov model.

The main aim is to make a correct consensus from noisy reads. 

## Install

1. Install [Rust](https://www.rust-lang.org/).
2. `git clone --recursive https://github.com/ban-m/kiley.git`
3. `cd kiley && cargo build --release` would create `kiley` binary under `./target/release`
4. `cargo test --release -- --nocapture` if you want.

## Lisence

MIT
