use rand::{rngs::StdRng, Rng, SeedableRng};
use rand_distr::Exp;

use structopt::StructOpt;

mod shoeshine_shop {
    use std::cmp::Reverse;
    use std::collections::{BinaryHeap, VecDeque};
    use std::convert::TryInto;

    use ordered_float::*;
    use rand::rngs::StdRng;
    use rand_distr::Distribution;

    #[derive(Copy, Clone, Debug, PartialEq, Ord, Eq, PartialOrd, enum_utils::TryFromRepr)]
    #[repr(usize)]
    enum Event {
        Arrived = 0,
        FirstFinished,
        SecondFinished,
    }

    #[derive(Copy, Clone, Debug, PartialEq, Ord, Eq, PartialOrd, enum_utils::TryFromRepr)]
    #[repr(usize)]
    enum State {
        Empty = 0,
        First,
        Second,
        Waiting,
        Both,
        Dropping,
        Invalid,
    }

    #[derive(Copy, Clone, Debug, Ord, Eq, PartialEq, PartialOrd)]
    struct Pair {
        time: OrderedFloat<f64>,
        event: Event,
    }

    #[rustfmt::skip]
    #[derive(Debug, Default)]
    struct Stats {
        state_counts:			[u32; State::Dropping as usize],
        state_counts_with_drop: [u32; State::Dropping as usize],

        time_in_state:			[f64; State::Dropping as usize],
        time_in_client:			[f64;  3],

        served_time:            f64,
        served_clients:         u32,
        
        arrived_clients:        u32,
        dropped_clients:        u32,
    }

    const STATE_TO_CLIENTS: [usize; State::Dropping as usize] = [0, 1, 1, 2, 2];

    #[rustfmt::skip]
    const EVENT_TO_STATE: [[State; 5]; 3] = [
        //                     EMPTY    FIRST     SECOND   WAITING   BOTH
        /* Arrived */         [First,   Dropping, Both,    Dropping, Dropping],
        /* First_Finished */  [Invalid, Second,   Invalid, Invalid,  Waiting],
        /* Second_Finished */ [Invalid, Invalid,  Empty,   Second,   First],
    ];

    macro_rules! report {
        ($title:expr, $counts:expr) => {{
            println!("{}", $title);
            let events: f64 = $counts.iter().copied().map(Into::<f64>::into).sum();

            for (i, count) in $counts.iter().enumerate() {
                let state: State = i.try_into().unwrap();
                println!("{:?}: {}", state, Into::<f64>::into(*count) / events);
            }

            println!();
        }};
    }

    pub struct Simulation<T: Distribution<f64>> {
        statistics: Stats,
        window: BinaryHeap<Reverse<Pair>>,
        iterations: u64,
        distributions: [T; 3],
        log_tail: u64,
    }

    use Event::*;
    use State::*;

    impl<T> Simulation<T>
    where
        T: Distribution<f64>,
    {
        pub fn new(
            arrival: T,
            first_serving: T,
            second_serving: T,
            iterations: u64,
        ) -> Simulation<T> {
            Simulation {
                statistics: Stats::default(),
                window: BinaryHeap::new(),
                iterations: iterations,
                distributions: [arrival, first_serving, second_serving],
                log_tail: 0,
            }
        }

        pub fn set_tail(&mut self, new_tail: u64) {
            self.log_tail = new_tail;
        }

        pub fn print_report(&mut self) {
            for (i, element) in self
                .statistics
                .state_counts_with_drop
                .iter_mut()
                .enumerate()
            {
                *element += self.statistics.state_counts[i];
            }

            report!("\ntime in states: ", self.statistics.time_in_state);
            report!("entries in states: ", self.statistics.state_counts);
            report!(
                "entries in states with dropouts: ",
                self.statistics.state_counts_with_drop
            );

            println!(
                "dropout: {}\naverage serving time: {}\naverage number of clients: {}",
                (self.statistics.dropped_clients as f64) / (self.statistics.arrived_clients as f64),
                self.statistics.served_time / (self.statistics.served_clients as f64),
                (self.statistics.time_in_client[1] + 2.0f64 * self.statistics.time_in_client[2])
                    / self.statistics.time_in_client.iter().sum::<f64>()
            );
        }

        pub fn simulate(&mut self, prng: &mut StdRng) -> bool {
            macro_rules! pusher {
                ($t:expr, $event:expr) => {{
                    let dt: f64 = self.distributions[$event as usize].sample(prng).into();
                    self.window.push(Reverse(Pair {
                        time: ($t + dt).into(),
                        event: $event,
                    }));
                }};
            }

            let mut prev = 0f64;
            let mut state = State::Empty;
            let mut arriving_times = VecDeque::<f64>::new();

            self.window.push(Reverse(Pair {
                time: 0.0.into(),
                event: Arrived,
            }));

            for i in 0..self.iterations {
                let event = self.window.pop().unwrap().0;
                if self.iterations - i < self.log_tail {
                    println!(
                        "{}: [{:?}] {:?} ==> [{:?}]",
                        event.time.0,
                        state,
                        event.event,
                        EVENT_TO_STATE[event.event as usize][state as usize]
                    );
                }
                match event.event {
                    Arrived => {
                        self.statistics.arrived_clients += 1;
                        pusher!(event.time.0, Arrived);
                    }
                    SecondFinished => {
                        self.statistics.served_time +=
                            event.time.0 - arriving_times.front().unwrap();
                        arriving_times.pop_front();
                        self.statistics.served_clients += 1;
                    }
                    _ => (),
                }
                let new_state = EVENT_TO_STATE[event.event as usize][state as usize];
                match new_state {
                    Invalid => return false,
                    Dropping => {
                        self.statistics.state_counts_with_drop[state as usize] += 1;
                        self.statistics.dropped_clients += 1;
                        continue;
                    }
                    First | Both if event.event == Arrived => {
                        arriving_times.push_back(event.time.0);
                        pusher!(event.time.0, FirstFinished);
                    }
                    Second => pusher!(event.time.0, SecondFinished),
                    _ => (),
                }
                self.statistics.time_in_state[state as usize] += event.time.0 - prev;
                self.statistics.time_in_client[STATE_TO_CLIENTS[state as usize]] +=
                    event.time.0 - prev;
                prev = event.time.0;
                state = new_state;
                self.statistics.state_counts[state as usize] += 1;
            }

            true
        }
    }
}

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

    ///expilictly set seed
    #[structopt(short)]
    seed: Option<u64>,

    ///change log tail
    #[structopt(short, default_value = "0")]
    tail: u64,
}

fn main() {
    let args = Args::from_args();

    let mut simulation = shoeshine_shop::Simulation::new(
        Exp::new(args.lambda).unwrap(),
        Exp::new(args.mu1).unwrap(),
        Exp::new(args.mu2).unwrap(),
        args.iterations * 1_000_000,
    );

    let seed = args.seed.unwrap_or(rand::thread_rng().gen());
    simulation.set_tail(args.tail);

    let mut prng: StdRng = SeedableRng::seed_from_u64(seed);

    if !simulation.simulate(&mut prng) {
        panic!("Error: invalid state reached, seed: {}", seed);
    }

    simulation.print_report();
}
