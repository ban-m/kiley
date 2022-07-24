//! Very thin Fasta reader. Only support batch IO.
use std::io::{BufRead, BufReader};
use std::io::{BufWriter, Write};
pub type FASTARecord = (String, Vec<u8>);
pub type FASTQRecord = (String, Vec<u8>, Vec<u8>);
/// Write records into the writer
pub fn write_fasta<W: Write>(
    wtr: &mut BufWriter<W>,
    records: &[FASTARecord],
) -> std::io::Result<()> {
    for (id, seq) in records {
        writeln!(wtr, ">{}\n{}", id, String::from_utf8_lossy(seq))?;
    }
    Ok(())
}

/// Write records into the writer.
pub fn write_fastq<W: Write>(
    wtr: &mut BufWriter<W>,
    records: &[FASTQRecord],
) -> std::io::Result<()> {
    for (id, seq, qual) in records {
        let seq = String::from_utf8_lossy(seq);
        let qual = String::from_utf8_lossy(qual);
        let split = format!("+{}", id);
        writeln!(wtr, "@{}\n{}\n{}\n{}", id, seq, split, qual)?;
    }
    Ok(())
}

/// Read file or stdin, return parsed fasta files.
/// If the record is mulformed, return empty vector.
pub fn read_fasta<P: AsRef<std::path::Path>>(
    file: &Option<P>,
) -> std::io::Result<Vec<FASTARecord>> {
    let stdin = std::io::stdin();
    let mut reader: Box<dyn BufRead> = match file {
        Some(file) => std::fs::File::open(file)
            .map(BufReader::new)
            .map(Box::new)?,
        None => {
            let lock = stdin.lock();
            Box::new(BufReader::new(lock))
        }
    };
    let mut contents = vec![];
    reader.read_to_end(&mut contents)?;
    Ok(parse_fasta(&contents))
}

fn parse_fasta(contents: &[u8]) -> Vec<FASTARecord> {
    let mut contents = contents.split(|&x| x == b'>');
    if let Some(first) = contents.next() {
        assert!(first.is_empty())
    }
    contents
        .filter_map(|record| {
            let mut record = record.splitn(2, |&x| x == b'\n');
            let id = record.next()?.split(|&x| x == b' ').next()?;
            let contents = record.next()?;
            let contents: Vec<_> = contents.iter().filter(|&&x| x != b'\n').copied().collect();
            Some((String::from_utf8_lossy(id).to_string(), contents))
        })
        .collect()
}

/// Read file or stdin, return parsed fastq files.
/// If the record is mulformed, return empty vector.
pub fn read_fastq<P: AsRef<std::path::Path>>(
    file: &Option<P>,
) -> std::io::Result<Vec<FASTQRecord>> {
    let stdin = std::io::stdin();
    let mut reader: Box<dyn BufRead> = match file {
        Some(file) => std::fs::File::open(file)
            .map(BufReader::new)
            .map(Box::new)?,
        None => {
            let lock = stdin.lock();
            Box::new(BufReader::new(lock))
        }
    };
    let mut contents = vec![];
    reader.read_to_end(&mut contents)?;
    Ok(parse_fastq(&contents))
}

fn parse_fastq(contents: &[u8]) -> Vec<FASTQRecord> {
    let mut contents = contents.split(|&x| x == b'\n');
    let parse = || -> Option<(String, Vec<u8>, Vec<u8>)> {
        let id = contents.next()?.split(|&x| x == b' ').next()?;
        let seq = contents.next()?.to_vec();
        let _id2 = contents.next()?;
        let qual = contents.next()?.to_vec();
        let id: String = String::from_utf8_lossy(&id[1..]).to_string();
        Some((id, seq, qual))
    };
    std::iter::from_fn(parse).collect()
}
