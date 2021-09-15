use clap::{App, Arg, SubCommand};
#[macro_use]
extern crate log;
fn subcommand_polish() -> App<'static, 'static> {
    SubCommand::with_name("polish")
        .version("0.1")
        .author("Bansho Masutani")
        .about("Polishing draft contigs by alignments.")
        .arg(
            Arg::with_name("verbose")
                .short("v")
                .multiple(true)
                .help("Debug mode"),
        )
        .arg(
            Arg::with_name("contigs")
                .long("contigs")
                .short("c")
                .value_name("FASTA")
                .takes_value(true)
                .required(true)
                .help("Draft contigs. FASTA format."),
        )
        .arg(
            Arg::with_name("reads")
                .long("reads")
                .short("r")
                .value_name("FASTA")
                .takes_value(true)
                .required(true)
                .help("Raw reads. FASTA format."),
        )
        .arg(
            Arg::with_name("alignments")
                .long("alignments")
                .value_name("SAM")
                .short("a")
                .takes_value(true)
                .required(true)
                .help("Alignment between [READS]-[CONTIGS]. SAM format."),
        )
        .arg(
            Arg::with_name("radius")
                .long("radius")
                .takes_value(true)
                .default_value("100")
                .help("Band width. Increase for erroneos reads."),
        )
        .arg(
            Arg::with_name("chunk_size")
                .long("chunk_size")
                .takes_value(true)
                .default_value("2025")
                .help("Length to polish at once. Increase for better QV."),
        )
        .arg(
            Arg::with_name("overlap")
                .long("overlap)")
                .takes_value(true)
                .default_value("25")
                .help("The length of overlapping of consective chunks."),
        )
        .arg(
            Arg::with_name("max_coverage")
                .long("max_coverage")
                .takes_value(true)
                .default_value("30")
                .help("Maximum coverage for each chunk."),
        )
        .arg(
            Arg::with_name("seed")
                .long("seed")
                .takes_value(true)
                .default_value("32389")
                .help("Seed"),
        )
        .arg(
            Arg::with_name("threads")
                .long("threads")
                .short("t")
                .takes_value(true)
                .default_value("1")
                .help("Number of threads"),
        )
}

fn subcommand_consensus() -> App<'static, 'static> {
    SubCommand::with_name("consensus")
        .version("0.1")
        .author("Bansho Masutani")
        .about("Taking a consensus from a set of reads from the same locus.")
        .arg(
            Arg::with_name("verbose")
                .short("v")
                .multiple(true)
                .help("Debug mode"),
        )
        .arg(
            Arg::with_name("reads")
                .long("reads")
                .short("r")
                .value_name("READS")
                .takes_value(true)
                .required(true)
                .help("Raw reads. FASTA format."),
        )
        .arg(
            Arg::with_name("radius")
                .long("radius")
                .takes_value(true)
                .default_value("100")
                .help("Band width. Increase for erroneos reads."),
        )
        .arg(
            Arg::with_name("seed")
                .long("seed")
                .takes_value(true)
                .default_value("32389")
                .help("Seed"),
        )
        .arg(
            Arg::with_name("repeat_num")
                .long("repeat_num")
                .takes_value(true)
                .default_value("10")
                .help("Repetition number."),
        )
        .arg(
            Arg::with_name("threads")
                .long("threads")
                .short("t")
                .takes_value(true)
                .default_value("1")
                .help("Number of threads"),
        )
}

fn polish(matches: &clap::ArgMatches) -> std::io::Result<()> {
    let contigs = matches
        .value_of("contigs")
        .and_then(|e| kiley::fasta::read_fasta(&Some(e)).ok())
        .unwrap();
    let reads = matches
        .value_of("reads")
        .and_then(|e| kiley::fasta::read_fasta(&Some(e)).ok())
        .unwrap();
    let alignments = matches
        .value_of("alignments")
        .and_then(|e| std::fs::File::open(e).ok().map(std::io::BufReader::new))
        .map(|rdr| kiley::sam::Sam::from_reader(rdr).records)
        .unwrap();
    let radius: usize = matches
        .value_of("radius")
        .and_then(|e| e.parse().ok())
        .unwrap();
    let chunk_size: usize = matches
        .value_of("chunk_size")
        .and_then(|e| e.parse().ok())
        .unwrap();
    let max_coverage: usize = matches
        .value_of("max_coverage")
        .and_then(|e| e.parse().ok())
        .unwrap();
    let overlap: usize = matches
        .value_of("overlap")
        .and_then(|e| e.parse().ok())
        .unwrap();
    let seed: u64 = matches
        .value_of("seed")
        .and_then(|e| e.parse().ok())
        .unwrap();
    use kiley::gphmm::*;
    let model = GPHMM::<Cond>::clr();
    let config =
        kiley::PolishConfig::with_model(radius, chunk_size, max_coverage, seed, overlap, model);
    let result = kiley::polish(&contigs, &reads, &alignments, &config);
    let stdout = std::io::stdout();
    let mut wtr = std::io::BufWriter::new(stdout.lock());
    kiley::fasta::write_fasta(&mut wtr, &result)
}

fn consensus(matches: &clap::ArgMatches) -> std::io::Result<()> {
    let reads = matches
        .value_of("reads")
        .and_then(|e| kiley::fasta::read_fasta(&Some(e)).ok())
        .unwrap();
    let radius: usize = matches
        .value_of("radius")
        .and_then(|e| e.parse().ok())
        .unwrap();
    let seed: u64 = matches
        .value_of("seed")
        .and_then(|e| e.parse().ok())
        .unwrap();
    let repeat_num: usize = matches
        .value_of("repeat_num")
        .and_then(|e| e.parse().ok())
        .unwrap();
    let seqs: Vec<_> = reads.iter().map(|x| x.1.as_slice()).collect();
    let polished = kiley::consensus(&seqs, seed, repeat_num, radius);
    println!(">0 Kiley consensus\n{}", String::from_utf8_lossy(&polished));
    Ok(())
}
fn main() -> std::io::Result<()> {
    let matches = App::new("kiley")
        .version("0.1")
        .author("Bansho Masutani")
        .about("Cosensus:[FASTA]->FASTA or Polish:[FASTA]x[FASTA]x[SAM]->[FASTA]")
        .setting(clap::AppSettings::ArgRequiredElseHelp)
        .subcommand(subcommand_consensus())
        .subcommand(subcommand_polish())
        .get_matches();
    if let Some(sub_m) = matches.subcommand().1 {
        let level = match sub_m.occurrences_of("verbose") {
            0 => "warn",
            1 => "info",
            2 => "debug",
            _ => "trace",
        };
        env_logger::Builder::from_env(env_logger::Env::default().default_filter_or(level)).init();
        let threads: usize = sub_m
            .value_of("threads")
            .and_then(|x| x.parse().ok())
            .unwrap();
        if let Err(why) = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build_global()
        {
            debug!("{:?} If you run `pipeline` module, this is Harmless.", why);
        }
    }
    debug!("Start");
    match matches.subcommand() {
        ("polish", Some(sub_m)) => polish(sub_m),
        ("consensus", Some(sub_m)) => consensus(sub_m),
        _ => unreachable!(),
    }
}
