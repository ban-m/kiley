use std::io::BufRead;

fn main() {
    let lines: Vec<_> = std::env::args()
        .nth(1)
        .and_then(|file| std::fs::File::open(file).ok())
        .map(std::io::BufReader::new)
        .unwrap()
        .lines()
        .filter_map(|line| line.ok())
        .collect();
    let mut refr = vec![];
    let mut ops = vec![];
    let mut query = vec![];
    for bucket in lines.chunks_exact(3) {
        ops.extend(
            bucket[0]
                .as_bytes()
                .iter()
                .zip(bucket[2].as_bytes().iter())
                .map(|(&x, &y)| {
                    if x == b' ' {
                        kiley::Op::Ins
                    } else if y == b' ' {
                        kiley::Op::Del
                    } else if x == y {
                        kiley::Op::Match
                    } else {
                        kiley::Op::Mismatch
                    }
                }),
        );
        refr.extend(bucket[0].as_bytes().iter().filter(|&&x| x != b' ').copied());
        query.extend(bucket[2].as_bytes().iter().filter(|&&x| x != b' ').copied());
    }
    let hmm = kiley::hmm::guided::PairHiddenMarkovModel::default();
    use kiley::hmm::guided::*;
    let template = refr;
    let seq = query;
    let radius = 60;
    let mut memory = Memory::with_capacity(template.len(), radius);
    let (refr, aln, query) = kiley::recover(&template, &seq, &ops);
    for ((refr, aln), query) in refr.chunks(150).zip(aln.chunks(150)).zip(query.chunks(150)) {
        eprintln!("{}", String::from_utf8_lossy(refr));
        eprintln!("{}", String::from_utf8_lossy(aln));
        eprintln!("{}", String::from_utf8_lossy(query));
    }
    eprintln!("------------------");
    let old_lk = hmm.eval_ln(&template, &seq, &ops);
    let lk = hmm.update_aln_path(&mut memory, &template, &seq, &mut ops);
    println!("{},{}", old_lk, lk);
    let (refr, aln, query) = kiley::recover(&template, &seq, &ops);
    for ((refr, aln), query) in refr.chunks(150).zip(aln.chunks(150)).zip(query.chunks(150)) {
        eprintln!("{}", String::from_utf8_lossy(refr));
        eprintln!("{}", String::from_utf8_lossy(aln));
        eprintln!("{}", String::from_utf8_lossy(query));
    }
}
