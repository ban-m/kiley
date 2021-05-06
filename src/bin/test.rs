// use kiley::bialignment::*;
fn main() -> std::io::Result<()> {
    env_logger::init();
    use std::io::*;
    let args: Vec<_> = std::env::args().collect();
    let inputs: Vec<Vec<_>> = std::fs::File::open(&args[1])
        .map(BufReader::new)?
        .lines()
        .filter_map(|x| x.ok())
        .filter(|x| !x.is_empty())
        .map(|x| x.bytes().collect())
        .collect();
    let draft = &inputs[0];
    let seqs = &inputs[1..];
    let config = kiley::PolishConfig::new(100, 2000, 30, 50, 43);
    let start = std::time::Instant::now();
    let consensus = kiley::bialignment::polish_until_converge_banded(&draft, seqs, 100);
    let middle = std::time::Instant::now();
    eprintln!("{}", (middle - start).as_millis());
    let consensus = kiley::polish_chunk_by_parts(&consensus, seqs, &config);
    let end = std::time::Instant::now();
    eprintln!("{}", (end - middle).as_millis());
    println!(">New\n{}", String::from_utf8_lossy(&consensus));
    // let consensus = kiley::polish_chunk_by_parts(&draft, seqs, &config);
    // println!(">Old\n{}", String::from_utf8_lossy(&consensus));
    Ok(())
}
