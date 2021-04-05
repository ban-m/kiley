//! Very thin Fasta reader. Only support batch IO.
use std::io;
use std::io::{BufRead, BufReader};
use std::io::{BufWriter, Write};
use std::path::Path;
/// Read file or stdin, return parsed fasta files.
/// If the record is mulformed, return empty vector.
pub fn read_fasta<P: AsRef<std::path::Path>>(
    file: &Option<P>,
) -> std::io::Result<Vec<(String, Vec<u8>)>> {
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

fn parse_fasta(contents: &[u8]) -> Vec<(String, Vec<u8>)> {
    vec![]
}

/// Read file or stdin, return parsed fastq files.
/// If the record is mulformed, return empty vector.
pub fn read_fastq<P: AsRef<std::path::Path>>(
    file: &Option<P>,
) -> std::io::Result<Vec<(String, Vec<u8>, Vec<u8>)>> {
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

fn parse_fastq(contents: &[u8]) -> Vec<(String, Vec<u8>, Vec<u8>)> {
    vec![]
}

#[derive(Debug)]
pub struct Reader<R: io::Read> {
    reader: BufReader<R>,
    line: String,
}

impl Reader<std::fs::File> {
    pub fn from_file<P: AsRef<Path>>(file: P) -> std::io::Result<Self> {
        let reader = std::fs::File::open(file).map(BufReader::new)?;
        let line = String::new();
        Ok(Self { reader, line })
    }
}

impl<R: io::Read> Reader<R> {
    pub fn new(reader: R) -> Self {
        let line = String::new();
        let reader = BufReader::new(reader);
        Self { reader, line }
    }
    pub fn read(&mut self, record: &mut Record) -> std::io::Result<usize> {
        record.clear();
        if self.line.is_empty() {
            self.reader.read_line(&mut self.line)?;
        }
        if self.line.is_empty() {
            Ok(1)
        } else if !self.line.starts_with('>') {
            Err(std::io::Error::from(std::io::ErrorKind::Other))
        } else {
            let mut header = self.line.split_whitespace();
            record.id = header.next().unwrap().trim_start_matches('>').to_string();
            record.desc = header.next().map(|e| e.to_string());
            while !self.line.is_empty() {
                self.line.clear();
                self.reader.read_line(&mut self.line).unwrap();
                if self.line.starts_with('>') {
                    break;
                } else {
                    record.seq.push_str(&self.line);
                }
            }
            Ok(1)
        }
    }
    pub fn records(self) -> Records<R> {
        Records { inner: self }
    }
}

#[derive(Debug)]
pub struct Records<R: io::Read> {
    inner: Reader<R>,
}

impl<R: io::Read> Iterator for Records<R> {
    type Item = std::io::Result<Record>;
    fn next(&mut self) -> Option<Self::Item> {
        let mut record = Record::default();
        let result = self.inner.read(&mut record);
        match result {
            Ok(_) if record.is_empty() => None,
            Ok(_) => Some(Ok(record)),
            Err(why) => Some(Err(why)),
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct Record {
    id: String,
    desc: Option<String>,
    seq: String,
}

impl Record {
    fn clear(&mut self) {
        self.id.clear();
        self.desc = None;
        self.seq.clear();
    }
    pub fn new() -> Self {
        Self::default()
    }
    pub fn with_data(id: &str, desc: &Option<String>, seq: &[u8]) -> Self {
        let id = id.to_string();
        let desc = desc.clone();
        let seq = String::from_utf8_lossy(seq).to_string();
        Self { id, desc, seq }
    }
    pub fn is_empty(&self) -> bool {
        self.id.is_empty() && self.seq.is_empty()
    }
    pub fn len(&self) -> usize {
        self.seq().len()
    }
    pub fn id(&self) -> &str {
        &self.id
    }
    pub fn seq(&self) -> &[u8] {
        self.seq.as_bytes()
    }
    pub fn desc(&self) -> Option<&String> {
        self.desc.as_ref()
    }
}

impl std::fmt::Display for Record {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        if let Some(ref desc) = self.desc {
            write!(f, ">{} {}\n{}", self.id, desc, self.seq)
        } else {
            write!(f, ">{}\n{}", self.id, self.seq)
        }
    }
}

/// Fastest method to open and parse fasta file.
pub fn parse_into_vec<P: AsRef<Path>>(file: P) -> std::io::Result<Vec<Record>> {
    let lines = std::fs::read_to_string(file)?;
    let mut result = Vec::with_capacity(bytecount::count(lines.as_bytes(), b'>'));
    let mut lines = lines.lines();
    let mut line = lines.next().unwrap();
    loop {
        let mut record = Record::default();
        let mut header = line[1..].splitn(2, ' ');
        record.id = header.next().unwrap().to_owned();
        record.desc = header.next().map(|e| e.to_owned());
        while let Some(next) = lines.next() {
            if next.starts_with('>') {
                line = next;
                break;
            } else {
                record.seq.push_str(next);
            }
        }
        if !record.seq.is_empty() {
            result.push(record);
        } else {
            return Ok(result);
        }
    }
}

pub fn parse_into_vec_from<R: io::Read>(reader: R) -> std::io::Result<Vec<Record>> {
    let mut lines = BufReader::new(reader).lines().filter_map(|e| e.ok());
    let mut result = Vec::with_capacity(10000);
    let mut line = lines.next().unwrap();
    loop {
        let mut record = Record::default();
        let mut header = line[1..].splitn(2, ' ');
        record.id = header.next().unwrap().to_owned();
        record.desc = header.next().map(|e| e.to_owned());
        while let Some(next) = lines.next() {
            if next.starts_with('>') {
                line = next;
                break;
            } else {
                record.seq.push_str(&next);
            }
        }
        if !record.seq.is_empty() {
            result.push(record);
        } else {
            return Ok(result);
        }
    }
}

#[derive(Debug)]
pub struct Writer<W: Write> {
    writer: BufWriter<W>,
}

impl<W: Write> Writer<W> {
    pub fn new(w: W) -> Self {
        Self {
            writer: BufWriter::new(w),
        }
    }
    pub fn write_record(&mut self, record: &Record) -> std::io::Result<()> {
        self.writer.write_all(b">")?;
        self.writer.write_all(record.id.as_bytes())?;
        if let Some(ref desc) = &record.desc {
            self.writer.write_all(b" ")?;
            self.writer.write_all(desc.as_bytes())?;
        }
        self.writer.write_all(b"\n")?;
        self.writer.write_all(record.seq.as_bytes())?;
        self.writer.write_all(b"\n")?;
        self.writer.flush()
    }
}
