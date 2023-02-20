fn main() {
    let size: usize = std::env::args().nth(1).unwrap().parse().unwrap();
    let xs: Vec<_> = vec![10; size];
    let sum: usize = xs.iter().sum();
    println!("{sum}");
}
