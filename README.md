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

## TODO

- If the alignment contains very long run of insertion/deletion at head/tail, the computing likelihood would be unstable in the guided PairHMM alignment. 
  This is because the alignment is global, and the leading runs of ins/del make the likelihood be exatly zero.
  Maybe we should fall back slow log-sum-exp version (current implementation is based on scaling as it is much faster and accurate for usual alignment....).

## Install

1. Install [Rust](https://www.rust-lang.org/).
2. `git clone --recursive https://github.com/ban-m/kiley.git`
3. `cd kiley && cargo build --release` would create `kiley` binary under `./target/release`
4. `cargo test --release -- --nocapture` if you want.

## Lisence

MIT
