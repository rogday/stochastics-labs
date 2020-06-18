use enum_map::enum_map;
use rand::{rngs::SmallRng, Rng, SeedableRng};
use rand_distr::Exp;
use structopt::StructOpt;

use shoeshine_shop::{statistics::enums::*, Simulation};

///shoe shine shop simulation
///
/// Shoe shine shop has two chairs, one for brushing (1) and another for polishing (2).
/// Customers arrive according to PP with rate λ, and enter only if first chair is empty.
/// Shoe-shiner takes exp(μ1) time for brushing and exp(μ2) time for polishing.
#[rustfmt::skip]
#[derive(StructOpt, Debug)]
#[structopt(name = "sim", setting = structopt::clap::AppSettings::AllowNegativeNumbers)]
struct Args {
    ///rate of customer arrival
    #[structopt(long, short)]
    lambda:     f64,

    ///rate of serving on the first chair
    #[structopt(long)]
    mu1:        f64,

    ///rate of serving on the second chair
    #[structopt(long)]
    mu2:        f64,

    ///millions of events to simulate
    #[structopt(short)]
    iterations: u64,

    ///explicitly set seed
    #[structopt(short)]
    seed:       Option<u64>,

    ///change log tail
    #[structopt(short, default_value = "0")]
    tail:       u64,
}

fn main() {
    let args = Args::from_args();

    assert!(args.iterations <= 1_000_000, "Number of iterations is too high.");
    assert!(args.iterations != 0, "Number of iterations can't be zero.");

    let iterations = args.iterations * 1_000_000;
    assert!(args.tail <= iterations, "Log length can't be greater than number of iterations");

    let simulation = Simulation {
        iterations,
        log_tail: args.tail,
        distributions: enum_map! {
            Event::Arrived        => Exp::new(args.lambda).expect("Exp(λ)"),
            Event::FirstFinished  => Exp::new(args.mu1).expect("Exp(μ1)"),
            Event::SecondFinished => Exp::new(args.mu2).expect("Exp(μ2)"),
        },
    };

    let seed = args.seed.unwrap_or_else(|| rand::thread_rng().gen());
    let mut prng: SmallRng = SeedableRng::seed_from_u64(seed);

    match simulation.simulate(&mut prng) {
        Ok(report) => println!("{}", report),
        Err(error) => eprintln!("Error: {:?}, seed: {}", error, seed),
    }
}
