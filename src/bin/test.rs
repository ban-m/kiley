use kiley::bialignment::guided::*;
fn main() {
    use rand::Rng;
    use rand::SeedableRng;
    use rand_xoshiro::Xoshiro256StarStar;
    const SEED: u64 = 1293890;
    let i = 0;
    // for i in 66..67 {
    {
        let radius = 40;
        let mut rng: Xoshiro256StarStar = SeedableRng::seed_from_u64(SEED + i);
        let xslen = rng.gen::<usize>() % 100 + 20;
        let xs = kiley::gen_seq::generate_seq(&mut rng, xslen);
        let prof = kiley::gen_seq::PROFILE;
        let ys = kiley::gen_seq::introduce_randomness(&xs, &mut rng, &prof);
        let (g_lk, g_banded_lk) = {
            use kiley::gphmm::*;
            let hmm = GPHMM::<Cond>::new_three_state(0.8, 0.05, 0.15, 0.95);
            let lk = hmm.likelihood(&xs, &ys);
            let b_lk = hmm.likelihood_banded(&xs, &ys, radius).unwrap();
            (lk, b_lk)
        };
        let (n_lk, n_banded_lk) = {
            let gap_output = [(4f64).recip(); 4];
            let (mat, mism) = (0.95, 0.05 / 3f64);
            let mut match_output = [[mism; 4]; 4];
            for i in 0..4 {
                match_output[i][i] = mat;
            }
            let quit_prob = 0.001;
            let hmm =
                kiley::hmm::PHMM::as_reversible(0.8, 0.15, &gap_output, &match_output, quit_prob);
            let naive = hmm.likelihood(&xs, &ys).1;
            let banded = hmm.likelihood_banded(&xs, &ys, radius).unwrap().1;
            (naive, banded)
        };
        let banded_lk = {
            let mat = (0.8, 0.1, 0.1);
            let ins = (0.8, 0.15, 0.05);
            let del = (0.85, 0.15);
            let mut emission = [0.05 / 3f64; 16];
            for i in 0..4 {
                emission[i * 4 + i] = 0.95;
            }
            let hmm = kiley::hmm::guided::PairHiddenMarkovModel::new(mat, ins, del, &emission);
            hmm.likelihood(&xs, &ys, radius)
        };
        eprintln!(
            "{}\t{}\t{}\t{}\t{}\t{}",
            i, g_lk, g_banded_lk, n_lk, n_banded_lk, banded_lk
        );
    }
}

// use std::io::{BufRead, BufReader};
// fn main() -> std::io::Result<()> {
//     env_logger::init();
//     let args: Vec<_> = std::env::args().collect();
//     let seqs: Vec<_> = std::fs::File::open(&args[1])
//         .map(BufReader::new)?
//         .lines()
//         .filter_map(|line| line.ok())
//         .map(|l| l.split('\t').nth(1).unwrap().to_string())
//         .collect();
//     let template = seqs[0].as_bytes();
//     let hmm = GPHMM::clr();
//     profile_multi_deletion_banded_check(&hmm, template, seqs[1].as_bytes(), 50);
//     Ok(())
// }

// fn profile_multi_deletion_banded_check<T: HMMType>(
//     model: &GPHMM<T>,
//     xs: &[u8],
//     ys: &[u8],
//     radius: isize,
// ) {
//     use banded::ProfileBanded;
//     use kiley::padseq::PadSeq;
//     // let xs: Vec<_> = xs.iter().rev().copied().take(350).collect();
//     // let ys: Vec<_> = ys.iter().rev().copied().take(350).collect();
//     let xs: Vec<_> = xs.iter().copied().take(350).collect();
//     let ys: Vec<_> = ys.iter().copied().take(350).collect();

//     // let (_, ops, _) = model.align(&xs, &ys);
//     // let (xa, oa, ya) = kiley::hmm::recover(&xs, &ys, &ops);
//     // for ((xa, oa), ya) in xa.chunks(200).zip(oa.chunks(200)).zip(ya.chunks(200)) {
//     //     println!("{}", String::from_utf8_lossy(xa));
//     //     println!("{}", String::from_utf8_lossy(oa));
//     //     println!("{}", String::from_utf8_lossy(ya));
//     // }
//     let orig_xs: Vec<_> = xs.to_vec();
//     let (xs, ys) = (PadSeq::new(xs), PadSeq::new(ys));
//     let profile = ProfileBanded::new(model, &xs, &ys, radius).unwrap();
//     let len = 6;
//     let difftable = profile.to_deletion_table(len);
//     // let n_profile = Profile::new(&model, &xs, &ys);
//     // let n_difftable = n_profile.to_deletion_table(len);
//     // for (i, (p, q)) in n_difftable.iter().zip(difftable.iter()).enumerate() {
//     //     assert!((p - q).abs() < 1f64, "{},{},{}", i, p, q);
//     // }
//     for (pos, diffs) in difftable.chunks(len - 1).enumerate() {
//         let mut xs: Vec<_> = orig_xs.clone();
//         xs.remove(pos);
//         for (i, lkd) in diffs.iter().enumerate() {
//             xs.remove(pos);
//             let xs = PadSeq::new(xs.as_slice());
//             let lk = model
//                 .likelihood_banded_inner(&xs, &ys, radius as usize)
//                 .unwrap();
//             assert!((lk - lkd).abs() < 10f64, "{},{},{},{}", lk, lkd, pos, i);
//         }
//     }
//     let copytable = profile.to_copy_table(len);
//     for (pos, copies) in copytable.chunks(len).enumerate() {
//         for (i, lkd) in copies.iter().enumerate() {
//             let xs: Vec<_> = orig_xs
//                 .iter()
//                 .take(pos + i)
//                 .chain(orig_xs.iter().skip(pos))
//                 .copied()
//                 .collect();
//             let xs = PadSeq::new(xs.as_slice());
//             let lk = model
//                 .likelihood_banded_inner(&xs, &ys, radius as usize)
//                 .unwrap();
//             assert!((lk - lkd).abs() < 10f64, "{},{},{},{}", lk, lkd, pos, i);
//         }
//     }
// }

// use rand::SeedableRng;
// fn main() {
//     env_logger::init();

// let seed = 231;
// let mut rng: rand_xoshiro::Xoroshiro128PlusPlus = SeedableRng::seed_from_u64(seed);
// use kiley::gen_seq;
// let prof = gen_seq::PROFILE.norm().mul(0.15);
// use kiley::gphmm::*;
// let hmm = GPHMM::clr();
// let len = 1000;
// let tandem_repeats = b"AT";
// let first_half = gen_seq::generate_seq(&mut rng, len);
// let second_half = gen_seq::generate_seq(&mut rng, len);
// let hap1 = {
//     let mut template = first_half.clone();
//     for _ in 0..10 {
//         template.extend_from_slice(tandem_repeats);
//     }
//     template.extend_from_slice(&second_half);
//     template
// };
// let hap2 = {
//     let mut template = first_half.clone();
//     for _ in 0..15 {
//         template.extend_from_slice(tandem_repeats);
//     }
//     template.extend_from_slice(&second_half);
//     template
// };
// let consensus = {
//     let mut template = first_half.clone();
//     for _ in 0..10 {
//         template.extend_from_slice(tandem_repeats);
//     }
//     template.extend_from_slice(&second_half);
//     template
// };
// let consensus = kiley::padseq::PadSeq::from(consensus);
// // let dellen = 6;
// let copylen = 6;
// for (i, seq) in (0..30).map(|i| (i, gen_seq::introduce_randomness(&hap1, &mut rng, &prof))) {
//     let query = kiley::padseq::PadSeq::from(seq);
//     let prof = kiley::gphmm::banded::ProfileBanded::new(&hmm, &consensus, &query, 200).unwrap();
//     let lk = prof.lk();
//     let copies = prof.to_copy_table(copylen);
//     for (idx, cs) in copies.iter().enumerate() {
//         let (pos, ci) = (idx / copylen, idx % copylen);
//         println!("{}\t{}\t{}\t{}", i, pos, ci, cs - lk);
//     }
//     // let dels = prof.to_copy_table(dellen);
//     // for (idx, ds) in dels.iter().enumerate() {
//     //     let (pos, di) = (idx / (dellen - 1), idx % (dellen - 1));
//     //     println!("{}\t{}\t{}\t{}", i, pos, di + 2, ds - lk);
//     // }
// }
// for (i, seq) in (30..60).map(|i| (i, gen_seq::introduce_randomness(&hap2, &mut rng, &prof))) {
//     let query = kiley::padseq::PadSeq::from(seq);
//     let prof = kiley::gphmm::banded::ProfileBanded::new(&hmm, &consensus, &query, 200).unwrap();
//     let lk = prof.lk();
//     let copies = prof.to_copy_table(copylen);
//     for (idx, cs) in copies.iter().enumerate() {
//         let (pos, ci) = (idx / copylen, idx % copylen);
//         println!("{}\t{}\t{}\t{}", i, pos, ci, cs - lk);
//     }
//     // let dels = prof.to_deletion_table(dellen);
//     // for (idx, ds) in dels.iter().enumerate() {
//     //     let (pos, di) = (idx / (dellen - 1), idx % (dellen - 1));
//     //     println!("{}\t{}\t{}\t{}", i, pos, di + 2, ds - lk);
//     // }
// }
// // let templates: Vec<_> = (10..15)
// //     .map(|time| {
// //         let mut template = first_half.clone();
// //         for _ in 0..time {
// //             template.extend_from_slice(tandem_repeats);
// //         }
// //         template.extend_from_slice(&second_half);
// //         template
// //     })
// //     .collect();
// // for seq in (0..30).map(|_| gen_seq::introduce_randomness(&hap1, &mut rng, &prof)) {
// //     let lks: Vec<_> = templates.iter().map(|t| hmm.likelihood(&t, &seq)).collect();
// //     let lks: Vec<_> = lks.iter().map(|x| format!("{:.3}", x - lks[3])).collect();
// //     println!("Hap1\t{}", lks.join("\t"));
// // }
// // for seq in (0..30).map(|_| gen_seq::introduce_randomness(&hap2, &mut rng, &prof)) {
// //     let lks: Vec<_> = templates.iter().map(|t| hmm.likelihood(&t, &seq)).collect();
// //     let lks: Vec<_> = lks.iter().map(|x| format!("{:.3}", x - lks[3])).collect();
// //     println!("Hap2\t{}", lks.join("\t"));
// // }
// }
