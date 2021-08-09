use rand::SeedableRng;
fn main() {
    env_logger::init();
    let seed = 231;
    let mut rng: rand_xoshiro::Xoroshiro128PlusPlus = SeedableRng::seed_from_u64(seed);
    use kiley::gen_seq;
    let prof = gen_seq::PROFILE.norm().mul(0.15);
    use kiley::gphmm::*;
    let hmm = GPHMM::clr();
    let len = 1000;
    let tandem_repeats = b"AT";
    let first_half = gen_seq::generate_seq(&mut rng, len);
    let second_half = gen_seq::generate_seq(&mut rng, len);
    let hap1 = {
        let mut template = first_half.clone();
        for _ in 0..10 {
            template.extend_from_slice(tandem_repeats);
        }
        template.extend_from_slice(&second_half);
        template
    };
    let hap2 = {
        let mut template = first_half.clone();
        for _ in 0..15 {
            template.extend_from_slice(tandem_repeats);
        }
        template.extend_from_slice(&second_half);
        template
    };
    let consensus = {
        let mut template = first_half.clone();
        for _ in 0..10 {
            template.extend_from_slice(tandem_repeats);
        }
        template.extend_from_slice(&second_half);
        template
    };
    let consensus = kiley::padseq::PadSeq::from(consensus);
    // let dellen = 6;
    let copylen = 6;
    for (i, seq) in (0..30).map(|i| (i, gen_seq::introduce_randomness(&hap1, &mut rng, &prof))) {
        let query = kiley::padseq::PadSeq::from(seq);
        let prof = kiley::gphmm::banded::ProfileBanded::new(&hmm, &consensus, &query, 200).unwrap();
        let lk = prof.lk();
        let copies = prof.to_copy_table(copylen);
        for (idx, cs) in copies.iter().enumerate() {
            let (pos, ci) = (idx / copylen, idx % copylen);
            println!("{}\t{}\t{}\t{}", i, pos, ci, cs - lk);
        }
        // let dels = prof.to_copy_table(dellen);
        // for (idx, ds) in dels.iter().enumerate() {
        //     let (pos, di) = (idx / (dellen - 1), idx % (dellen - 1));
        //     println!("{}\t{}\t{}\t{}", i, pos, di + 2, ds - lk);
        // }
    }
    for (i, seq) in (30..60).map(|i| (i, gen_seq::introduce_randomness(&hap2, &mut rng, &prof))) {
        let query = kiley::padseq::PadSeq::from(seq);
        let prof = kiley::gphmm::banded::ProfileBanded::new(&hmm, &consensus, &query, 200).unwrap();
        let lk = prof.lk();
        let copies = prof.to_copy_table(copylen);
        for (idx, cs) in copies.iter().enumerate() {
            let (pos, ci) = (idx / copylen, idx % copylen);
            println!("{}\t{}\t{}\t{}", i, pos, ci, cs - lk);
        }
        // let dels = prof.to_deletion_table(dellen);
        // for (idx, ds) in dels.iter().enumerate() {
        //     let (pos, di) = (idx / (dellen - 1), idx % (dellen - 1));
        //     println!("{}\t{}\t{}\t{}", i, pos, di + 2, ds - lk);
        // }
    }
    // let templates: Vec<_> = (10..15)
    //     .map(|time| {
    //         let mut template = first_half.clone();
    //         for _ in 0..time {
    //             template.extend_from_slice(tandem_repeats);
    //         }
    //         template.extend_from_slice(&second_half);
    //         template
    //     })
    //     .collect();
    // for seq in (0..30).map(|_| gen_seq::introduce_randomness(&hap1, &mut rng, &prof)) {
    //     let lks: Vec<_> = templates.iter().map(|t| hmm.likelihood(&t, &seq)).collect();
    //     let lks: Vec<_> = lks.iter().map(|x| format!("{:.3}", x - lks[3])).collect();
    //     println!("Hap1\t{}", lks.join("\t"));
    // }
    // for seq in (0..30).map(|_| gen_seq::introduce_randomness(&hap2, &mut rng, &prof)) {
    //     let lks: Vec<_> = templates.iter().map(|t| hmm.likelihood(&t, &seq)).collect();
    //     let lks: Vec<_> = lks.iter().map(|x| format!("{:.3}", x - lks[3])).collect();
    //     println!("Hap2\t{}", lks.join("\t"));
    // }
}
