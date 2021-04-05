use clap::{App, Arg};
use kiley::consensus;
fn main() -> std::io::Result<()> {
    let matches = App::new("kiley")
        .version("0.1")
        .author("Bansho Masutani")
        .about("Raw Read(FASTA)->Consensus(TEXT)")
        .setting(clap::AppSettings::ArgRequiredElseHelp)
        .arg(
            Arg::with_name("verbose")
                .short("v")
                .multiple(true)
                .help("Debug mode"),
        )
        .arg(
            Arg::with_name("read_type")
                .long("read_type")
                .takes_value(true)
                .default_value(&"CLR")
                .possible_values(&["CCS", "CLR", "ONT"])
                .help("Read type. CCS, CLR, or ONT."),
        )
        .arg(
            Arg::with_name("seed")
                .long("seed")
                .takes_value(true)
                .default_value(&"32389")
                .help("Seed"),
        )
        .arg(
            Arg::with_name("subchunk_size")
                .long("subchunk_size")
                .takes_value(true)
                .default_value(&"5")
                .help("Size of a subchunk"),
        )
        .arg(
            Arg::with_name("repeat_num")
                .long("repeat_num")
                .takes_value(true)
                .default_value(&"10")
                .help("Repetition number."),
        )
        .get_matches();
    let seed: u64 = matches
        .value_of("seed")
        .and_then(|e| e.parse().ok())
        .unwrap();
    let _subchunk_size: usize = matches
        .value_of("subchunk_size")
        .and_then(|e| e.parse().ok())
        .unwrap();
    let repeat_num: usize = matches
        .value_of("repeat_num")
        .and_then(|e| e.parse().ok())
        .unwrap();
    Ok(())
}
