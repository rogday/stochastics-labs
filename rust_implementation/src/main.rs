use rand::{rngs::StdRng, Rng, SeedableRng};
use rand_distr::Exp;

use enum_map::enum_map;
use structopt::StructOpt;

mod shoeshine_shop {
    use enum_map::{Enum, EnumMap};

    //use std::cmp::Reverse;
    //use std::collections::BinaryHeap;

    use either::*;

    // heapless is too slow for some reason
    //use heapless::binary_heap::{BinaryHeap, Min};
    //use heapless::consts::*;
    use itertools::*;
    use ordered_float::*; // type level integer used to specify capacity

    use rand::rngs::StdRng;
    use rand_distr::Distribution;

    /// Wrapper around an array of two elements
    #[derive(Default)]
    struct Queue<T: Copy> {
        buffer: [T; 2],
        end: usize,
    }

    impl<T: Copy> Queue<T> {
        fn pop_front(&mut self) -> T {
            self.end -= 1;
            self.buffer.swap(0, 1);
            self.buffer[1]
        }
        fn push_back(&mut self, val: T) {
            self.buffer[self.end] = val;
            self.end += 1;
        }
    }

    #[derive(Default)]
    struct TreeMin3<T> {
        buffer: [T; 3],
        len: usize,
    }

    impl<T: PartialOrd + Copy + Default + std::fmt::Debug> TreeMin3<T> {
        fn new() -> Self {
            TreeMin3 {
                buffer: [T::default(); 3],
                len: 0,
            }
        }

        fn push(&mut self, val: T) {
            let mut i = 0;
            while i < self.len && val > self.buffer[i] {
                i += 1;
            }

            let mut k = self.len;
            while k > i {
                self.buffer.swap(k, k - 1);
                k -= 1;
            }

            self.buffer[i] = val;
            self.len += 1;
        }

        fn pop(&mut self) -> T {
            for k in 0..self.len - 1 {
                self.buffer.swap(k, k + 1);
            }

            self.len -= 1;
            self.buffer[self.len]
        }
    }

    #[derive(Enum, Debug, Copy, Clone, PartialEq, Ord, Eq, PartialOrd, Hash)]
    #[repr(usize)]
    pub enum Event {
        Arrived,
        FirstFinished,
        SecondFinished,
    }

    impl Default for Event {
        fn default() -> Self {
            Event::SecondFinished
        }
    }

    #[derive(Enum, Debug, Copy, Clone, PartialEq, Ord, Eq, PartialOrd, Hash)]
    #[repr(usize)]
    pub enum State {
        Empty,
        First,
        Second,
        Waiting,
        Both,
    }
    #[derive(Debug)]
    enum Transition {
        Dropping,
    }

    #[derive(Default, Copy, Clone, Debug, Ord, Eq, PartialEq, PartialOrd)]
    struct Pair {
        time: OrderedFloat<f64>,
        event: Event,
    }

    #[rustfmt::skip]
    #[derive(Debug, Default)]
    struct Stats {
        /// How many times system was in state S
        counts:	        EnumMap<State, u32>,

        /// How many times system dropped client in state S
        drops:	        EnumMap<State, u32>,

        /// Time spent in state S
        t_state:        EnumMap<State, f64>,

        /// How long system was in serving state (From First to SecondFinished)
        served_time:    f64,

        /// How many clients were served
        served_clients: u32,

        /// How many clients arrived including dropped
        arrived:        u32,
    }

    pub struct Report {
        pub t_states: EnumMap<State, f64>,
        pub counts: EnumMap<State, f64>,
        pub dropful_counts: EnumMap<State, f64>,

        pub dropout: f64,
        pub t_serving_avg: f64,
        pub n_clients_avg: f64,
    }

    /// Map state to number of clients in that state
    // Not a hasher because otherwise Eq must be reimplemented as well
    fn client_number(state: State) -> usize {
        match state {
            State::Empty => 0,
            State::First | State::Second => 1,
            State::Waiting | State::Both => 2,
        }
    }

    /// Determine new state or pseudostate(Transition) from current state and incoming event
    fn advance(state: State, event: Event) -> Result<Either<State, Transition>, &'static str> {
        use State::*;
        use Transition::*;

        // explicit matching to ensure compile time error in case of newly added state
        match event {
            Event::Arrived => Ok(match state {
                Empty => Left(First),
                Second => Left(Both),
                // first chair is occupied
                First | Waiting | Both => Right(Dropping),
            }),
            Event::FirstFinished => match state {
                First => Ok(Left(Second)),
                Both => Ok(Left(Waiting)),
                // first chair is empty/already finished
                Empty | Second | Waiting => Err("Invalid state reached"),
            },
            Event::SecondFinished => match state {
                Second => Ok(Left(Empty)),
                Waiting => Ok(Left(Second)),
                Both => Ok(Left(First)),
                // second chair is empty
                Empty | First => Err("Invalid state reached"),
            },
        }
    }

    fn report<T>(counts: &EnumMap<State, T>) -> EnumMap<State, f64>
    where
        T: Copy + Into<f64>,
    {
        let mut res = EnumMap::new();
        let sum: f64 = counts.values().copied().map_into::<f64>().sum();

        for (state, count) in counts.iter() {
            let normalized: f64 = (*count).into() / sum;
            res[state] = normalized;
        }

        res
    }

    /// Print normalized table-like report for all states
    fn print_report(title: &str, counts: &EnumMap<State, f64>) {
        println!("{}", title);

        for (state, value) in counts.iter() {
            println!("{:?}:   \t{}", state, value);
        }

        println!();
    }

    pub struct Simulation<T: Distribution<f64>> {
        stats: Stats,
        window: TreeMin3<Pair>,
        // window: BinaryHeap<Pair, U3, Min>,
        iterations: u64,
        distributions: EnumMap<Event, T>,
        log_tail: u64,
    }

    impl<T> Simulation<T>
    where
        T: Distribution<f64>,
    {
        pub fn new(distributions: EnumMap<Event, T>, iterations: u64) -> Simulation<T> {
            assert!(distributions.len() == 3);
            Simulation {
                stats: Stats::default(),
                window: TreeMin3::new(),
                // window: BinaryHeap::new(),
                iterations,
                distributions,
                log_tail: 0,
            }
        }

        pub fn set_tail(&mut self, new_tail: u64) {
            self.log_tail = new_tail;
        }

        pub fn full_report(&self) -> Report {
            let mut dropful_counts = EnumMap::<State, u32>::new();

            for (state, count) in self.stats.counts.iter() {
                dropful_counts[state] = count + self.stats.drops[state];
            }

            // How long there was {0, 1, 2} clients in the system
            let mut t_client = [0f64; 3];
            for (state, time) in self.stats.t_state.iter() {
                t_client[client_number(state)] += time
            }

            let dropped: u32 = self.stats.drops.values().sum();

            Report {
                t_states: report(&self.stats.t_state),
                counts: report(&self.stats.counts),
                dropful_counts: report(&dropful_counts),

                dropout: (dropped as f64) / (self.stats.arrived as f64),
                t_serving_avg: self.stats.served_time / (self.stats.served_clients as f64),
                n_clients_avg: (t_client[1] + 2.0f64 * t_client[2]) / t_client.iter().sum::<f64>(),
            }
        }

        /// Report of collected stats
        pub fn print_full_report(&self) {
            let report = self.full_report();

            print_report("\nTime in states: ", &report.t_states);
            print_report("Entries in states: ", &report.counts);
            print_report("Entries in states with dropouts: ", &report.dropful_counts);

            println!(
                "Dropout:                   {}\n\
                 Average serving time:      {}\n\
                 Average number of clients: {}",
                report.dropout, report.t_serving_avg, report.n_clients_avg
            );
        }

        pub fn simulate(&mut self, prng: &mut StdRng) -> Result<(), &str> {
            // generate dt and insert (from_time + dt, event) in a window
            macro_rules! pusher {
                ($t:expr, $event:expr) => {{
                    let dt: f64 = self.distributions[$event].sample(prng).into();

                    self.window.push(Pair {
                        time: ($t + dt).into(),
                        event: $event,
                    });
                }};
            }

            // time of last change of state
            let mut prev = 0f64;
            let mut state = State::Empty;
            // basically two floats on stack
            let mut arriving_times = Queue::<f64>::default();

            self.window.push(Pair {
                time: 0.0.into(),
                event: Event::Arrived,
            });

            // get current event, resubscribe it if needed, determine new state,
            // generate new events if we got a state and not a transition
            // collect statistics everywhere
            for i in 0..self.iterations {
                let current = self.window.pop();
                let new_state = advance(state, current.event)?;

                if self.iterations - i < self.log_tail + 1 {
                    println!(
                        "{:.10}: [{:?}] {:?} ==> [{:?}]",
                        i, state, current.event, new_state
                    );
                }
                match current.event {
                    Event::Arrived => {
                        self.stats.arrived += 1;
                        pusher!(current.time.0, Event::Arrived);
                    }
                    Event::SecondFinished => {
                        self.stats.served_time += current.time.0 - arriving_times.pop_front();
                        self.stats.served_clients += 1;
                    }
                    _ => (),
                }

                match new_state {
                    Right(Transition::Dropping) => {
                        self.stats.drops[state] += 1;
                        continue;
                    }
                    Left(_) if current.event == Event::Arrived => {
                        arriving_times.push_back(current.time.0);
                        pusher!(current.time.0, Event::FirstFinished);
                    }
                    Left(State::Second) => pusher!(current.time.0, Event::SecondFinished),
                    _ => (),
                }
                self.stats.t_state[state] += current.time.0 - prev;

                prev = current.time.0;
                state = new_state.left().unwrap();

                self.stats.counts[state] += 1;
            }

            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::shoeshine_shop::*;

    use enum_map::{enum_map, EnumMap};
    use rand::{rngs::StdRng, Rng, SeedableRng};
    use rand_distr::Exp;

    const EPS: f64 = 0.01;
    const ITERATIONS: u64 = 50_000_000;

    fn run(lambda: f64, mu1: f64, mu2: f64) -> Report {
        let distributions = enum_map! {
            Event::Arrived        => Exp::new(lambda).unwrap(),
            Event::FirstFinished  => Exp::new(mu1).unwrap(),
            Event::SecondFinished => Exp::new(mu2).unwrap(),
        };

        let mut simulation = Simulation::new(distributions, ITERATIONS);

        // NOTE: that's not lazy
        let seed = rand::thread_rng().gen();

        let mut prng: StdRng = SeedableRng::seed_from_u64(seed);

        if let Err(error) = simulation.simulate(&mut prng) {
            panic!("Error: {}, seed: {}", error, seed);
        }

        simulation.full_report()
    }

    fn check(map: &EnumMap<State, f64>, v: &[f64]) {
        for (i, value) in map.values().enumerate() {
            assert!(value - v[i] < EPS, format!("Error in state {}", i));
        }
    }

    #[test]
    fn case_one() {
        let report = run(3.0, 20.0, 1.0);

        let t_states = vec![0.758597, 0.0130194, 0.227571, 0.650963, 0.0325863];
        let counts = vec![0.0833838, 0.0952409, 0.333333, 0.238093, 0.249949];
        let dropful_counts = vec![0.0472291, 0.0620404, 0.188802, 0.540162, 0.161766];

        check(&report.t_states, &t_states);
        check(&report.counts, &counts);
        check(&report.dropful_counts, &dropful_counts);

        assert!(report.dropout - 0.696653 < EPS);
        assert!(report.t_serving_avg - 1.76369 < EPS);
        assert!(report.n_clients_avg - 1.60759 < EPS);
    }

    #[test]
    fn case_two() {
        let report = run(1.0, 1.0, 1.0);

        let t_states = vec![0.223083, 0.333209, 0.222046, 0.11069, 0.110972];
        let counts = vec![0.166769, 0.24998, 0.333334, 0.0833531, 0.166564];
        let dropful_counts = vec![0.11776, 0.352968, 0.235376, 0.117753, 0.176142];

        check(&report.t_states, &t_states);
        check(&report.counts, &counts);
        check(&report.dropful_counts, &dropful_counts);

        assert!(report.dropout - 0.555263 < EPS);
        assert!(report.t_serving_avg - 2.24861 < EPS);
        assert!(report.n_clients_avg - 0.998578 < EPS);
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

    let mut simulation = Simulation::new(distributions, args.iterations * 1_000_000);

    // NOTE: that's not lazy
    let seed = args.seed.unwrap_or(rand::thread_rng().gen());
    simulation.set_tail(args.tail);

    let mut prng: StdRng = SeedableRng::seed_from_u64(seed);

    if let Err(error) = simulation.simulate(&mut prng) {
        panic!("Error: {}, seed: {}", error, seed);
    }

    simulation.print_full_report();
}
