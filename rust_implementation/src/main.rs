use rand::{rngs::SmallRng, Rng, SeedableRng};

use rand_distr::Exp;

use enum_map::enum_map;
use structopt::StructOpt;

mod shoeshine_shop {
    use enum_map::{Enum, EnumMap};

    use either::*;

    use itertools::*;
    use ordered_float::*;

    use rand::rngs::SmallRng;
    use rand_distr::Distribution;

    // TODO: get rid of this
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

    // TODO: get rid of this
    /// Sorted array on stack with capacity 3
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

    #[derive(Debug)]
    pub enum SimulationError {
        InvalidState,
    }

    #[derive(Default, Copy, Clone, Debug, Ord, Eq, PartialEq, PartialOrd)]
    struct Pair {
        time: OrderedFloat<f64>,
        event: Event,
    }

    #[rustfmt::skip]
    #[derive(Debug, Default)]
    struct Stats {
        /// How many times the system was in the state S
        counts:	        EnumMap<State, u32>,

        /// How many times the system dropped client in the state S
        drops:	        EnumMap<State, u32>,

        /// Time spent in the state S
        t_state:        EnumMap<State, f64>,

        /// How long the system was in serving state (From First to SecondFinished)
        served_time:    f64,

        /// How many clients were served
        served_clients: u32,

        /// How many clients arrived including dropped
        arrived:        u32,
    }

    /// Basically normalized Stats
    pub struct Report {
        /// Time spent in the state S, *Normalized*
        pub t_states: EnumMap<State, f64>,

        /// How many times the system was in the state S, *Normalized*
        pub counts: EnumMap<State, f64>,

        /// How many times the system dropped client in the state S, *Normalized*
        pub dropful_counts: EnumMap<State, f64>,

        /// Ratio of clients that walked in during busy state to all arrived clients
        pub dropout: f64,

        /// Average serving time (from Arrived to SecondFinished)
        pub t_serving_avg: f64,

        /// Average number of clients in the system
        pub n_clients_avg: f64,
    }

    impl Report {
        /// Print normalized table-like report for all states
        fn print_report(
            f: &mut std::fmt::Formatter<'_>,
            title: &str,
            counts: &EnumMap<State, f64>,
        ) -> std::fmt::Result {
            writeln!(f, "{}", title)?;

            for (state, value) in counts.iter() {
                writeln!(f, "{:?}:   \t{}", state, value)?;
            }

            writeln!(f)
        }
    }

    impl std::fmt::Display for Report {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            Report::print_report(f, "\nTime in states: ", &self.t_states)?;
            Report::print_report(f, "Entries in states: ", &self.counts)?;
            Report::print_report(f, "Entries in states with dropouts: ", &self.dropful_counts)?;

            writeln!(
                f,
                "Dropout:                   {}\n\
                 Average serving time:      {}\n\
                 Average number of clients: {}",
                self.dropout, self.t_serving_avg, self.n_clients_avg
            )
        }
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
    fn advance(state: State, event: Event) -> Result<Either<State, Transition>, SimulationError> {
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
                Empty | Second | Waiting => Err(SimulationError::InvalidState),
            },
            Event::SecondFinished => match state {
                Second => Ok(Left(Empty)),
                Waiting => Ok(Left(Second)),
                Both => Ok(Left(First)),
                // second chair is empty
                Empty | First => Err(SimulationError::InvalidState),
            },
        }
    }

    /// Divide each record by sum of all records
    fn normalized<T>(counts: &EnumMap<State, T>) -> EnumMap<State, f64>
    where
        T: Copy + Into<f64>,
    {
        let mut res = EnumMap::new();
        let sum: f64 = counts.values().copied().map_into::<f64>().sum();

        for (state, &count) in counts.iter() {
            let normalized: f64 = count.into() / sum;
            res[state] = normalized;
        }

        res
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
            // no assertions on distributions needed because EnumMap elements are stored in array and therefore always initialized
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

        /// Transform collected data into useful statistics
        fn form_report(&self) -> Report {
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
                t_states: normalized(&self.stats.t_state),
                counts: normalized(&self.stats.counts),
                dropful_counts: normalized(&dropful_counts),

                dropout: (dropped as f64) / (self.stats.arrived as f64),
                t_serving_avg: self.stats.served_time / (self.stats.served_clients as f64),
                n_clients_avg: (t_client[1] + 2.0f64 * t_client[2]) / t_client.iter().sum::<f64>(),
            }
        }

        pub fn simulate(&mut self, prng: &mut SmallRng) -> Result<Report, SimulationError> {
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

            Ok(self.form_report())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::shoeshine_shop::*;

    use enum_map::{enum_map, EnumMap};
    use rand::{rngs::SmallRng, Rng, SeedableRng};
    use rand_distr::Exp;

    // release
    #[cfg(not(debug_assertions))]
    const EPS: f64 = 0.001;
    #[cfg(not(debug_assertions))]
    const ITERATIONS: u64 = 50_000_000;

    // debug
    #[cfg(debug_assertions)]
    const EPS: f64 = 0.01;
    #[cfg(debug_assertions)]
    const ITERATIONS: u64 = 1_000_000;

    fn run(lambda: f64, mu1: f64, mu2: f64) -> Report {
        let distributions = enum_map! {
            Event::Arrived        => Exp::new(lambda).unwrap(),
            Event::FirstFinished  => Exp::new(mu1).unwrap(),
            Event::SecondFinished => Exp::new(mu2).unwrap(),
        };

        let mut simulation = Simulation::new(distributions, ITERATIONS);

        let seed = rand::thread_rng().gen();
        let mut prng: SmallRng = SeedableRng::seed_from_u64(seed);

        // if the test fails, it will be printed out
        println!("Using seed: {}", seed);

        simulation.simulate(&mut prng).expect("Simulation failed")
    }

    fn approx_eq_assert(a: f64, b: f64, msg: &str) {
        assert!(
            (a - b).abs() < EPS,
            format!("{}: expected {}, got {}", msg, b, a)
        );
    }

    fn assert_maps(title: &str, map: &EnumMap<State, f64>, v: &EnumMap<State, f64>) {
        println!("Checking \"{}\"...", title);
        for (state, &value) in map.iter() {
            approx_eq_assert(value, v[state], &format!("Wrong value in {:?}", state));
        }
    }

    fn tester(reference: &Report, actual: &Report) {
        assert_maps("Time", &actual.t_states, &reference.t_states);
        assert_maps("Counts", &actual.counts, &reference.counts);
        assert_maps(
            "Counts with drops",
            &actual.dropful_counts,
            &reference.dropful_counts,
        );

        approx_eq_assert(actual.dropout, reference.dropout, "Dropout is wrong");

        approx_eq_assert(
            actual.t_serving_avg,
            reference.t_serving_avg,
            "Avg. serving time is wrong",
        );

        approx_eq_assert(
            actual.n_clients_avg,
            reference.n_clients_avg,
            "Avg. n of clients is wrong",
        );
    }

    #[test]
    fn case_one() {
        let reference = Report {
            t_states: enum_map! {
                State::Empty   => 0.07597501619196396,
                State::First   => 0.013018780028126743,
                State::Second  => 0.22780664536179357,
                State::Waiting => 0.6506552969310084,
                State::Both    => 0.032544261487107304,
            },

            counts: enum_map! {
                State::Empty   => 0.08335626064523295,
                State::First   => 0.09526323676903557,
                State::Second  => 0.3333333333333333,
                State::Waiting => 0.23807009656429776,
                State::Both    => 0.24997707268810038,
            },

            dropful_counts: enum_map! {
                State::Empty   => 0.0472456,
                State::First   => 0.06209351,
                State::Second  => 0.18893042,
                State::Waiting => 0.53980918,
                State::Both    => 0.16192129,
            },

            dropout: 0.6963212748684511,
            t_serving_avg: 1.7641820719255652,
            n_clients_avg: 1.6072245422261517,
        };

        let actual = run(3.0, 20.0, 1.0);

        tester(&reference, &actual);
    }

    #[test]
    fn case_two() {
        let reference = Report {
            t_states: enum_map! {
                State::Empty   => 0.22211660968919172,
                State::First   => 0.33346805672671304,
                State::Second  => 0.22218707296352663,
                State::Waiting => 0.11115295878727308,
                State::Both    => 0.11107530183329563,
            },

            counts: enum_map! {
                State::Empty   => 0.16666983570972363,
                State::First   => 0.25001529500140085,
                State::Second  => 0.3333333380561993,
                State::Waiting => 0.08331804305479844,
                State::Both    => 0.1666634881778778,
            },

            dropful_counts: enum_map! {
                State::Empty   => 0.11763326,
                State::First   => 0.35297929,
                State::Second  => 0.23526205,
                State::Waiting => 0.11770195,
                State::Both    => 0.17642345,
            },

            dropout: 0.555669964286005,
            t_serving_avg: 2.250242037846493,
            n_clients_avg: 1.000111650931377,
        };

        let actual = run(1.0, 1.0, 1.0);

        tester(&reference, &actual);
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

    let seed = args.seed.unwrap_or_else(|| rand::thread_rng().gen());
    simulation.set_tail(args.tail);

    let mut prng: SmallRng = SeedableRng::seed_from_u64(seed);

    match simulation.simulate(&mut prng) {
        Ok(report) => println!("{}", report),
        Err(error) => panic!("Error: {:?}, seed: {}", error, seed),
    }
}
