use rand::{rngs::SmallRng, Rng, SeedableRng};

use rand_distr::Exp;

use enum_map::enum_map;
use structopt::StructOpt;

mod shoeshine_shop;

///shoe shine shop simulation
///
/// Shoe shine shop has two chairs, one for brushing (1) and another for polishing (2).
/// Customers arrive according to PP with rate λ, and enter only if first chair is empty.
/// Shoe-shiners takes exp(μ1) time for brushing and exp(μ2) time for polishing.
#[derive(StructOpt, Debug)]
#[structopt(name = "sim")]
struct Args {
    ///rate of customer arrival
    #[structopt(long)]
    lambda: f64,

    ///rate of serving on the first chair
    #[structopt(long)]
    mu1: f64,

    ///rate of serving on the second chair
    #[structopt(long)]
    mu2: f64,

    ///millions of events to simulate
    #[structopt(short)]
    iterations: u64,

    ///explicitly set seed
    #[structopt(short)]
    seed: Option<u64>,

    ///change log tail
    #[structopt(short, default_value = "0")]
    tail: u64,
}

fn main() {
    use shoeshine_shop::*;
    let args = Args::from_args();

    let distributions = enum_map! {
        Event::Arrived        => Exp::new(args.lambda).unwrap(),
        Event::FirstFinished  => Exp::new(args.mu1).unwrap(),
        Event::SecondFinished => Exp::new(args.mu2).unwrap(),
    };

    let mut simulation = Simulation::new(distributions, args.iterations * 1_000_000, args.tail);

    let seed = args.seed.unwrap_or_else(|| rand::thread_rng().gen());

    let mut prng: SmallRng = SeedableRng::seed_from_u64(seed);

    match simulation.simulate(&mut prng) {
        Ok(report) => println!("{}", report),
        Err(error) => panic!("Error: {:?}, seed: {}", error, seed),
    }
}
