# Kiley -- Consensus module

Author: Bansho Masutani

Email: ban-m@g.ecc.u-tokyo.ac.jp

## Install

1. Install [Rust](https://www.rust-lang.org/).
2. `git clone https://github.com/ban-m/kiley.git`
3. `cd kiley && cargo build --release` would create `kiley` binary under `./target/release`
4. `cargo test --release -- --nocapture` if you want.


## Synopsis
```
cat ${FASTA} | ./target/release/kieley > ${SEQ}
```
It writes a consensus sequence into stdout as a UTF8-encoded text.