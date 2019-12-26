use rand::{rngs::StdRng, Rng, SeedableRng};
use rand_distr::Exp;

use structopt::StructOpt;

use fnv::FnvHashMap;

mod shoeshine_shop {
    use std::cmp::Reverse;
    use std::collections::BinaryHeap;

    use either::*;
    use fnv::FnvHashMap; //standard hasher is DDOS-resistant and therefore slow-ish
    use itertools::*;
    use ordered_float::*;

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

    #[derive(Debug, Copy, Clone, PartialEq, Ord, Eq, PartialOrd, Hash)]
    #[repr(usize)]
    pub enum Event {
        Arrived,
        FirstFinished,
        SecondFinished,
    }

    #[derive(Debug, Copy, Clone, PartialEq, Ord, Eq, PartialOrd, Hash)]
    #[repr(usize)]
    enum State {
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

    #[derive(Copy, Clone, Debug, Ord, Eq, PartialEq, PartialOrd)]
    struct Pair {
        time: OrderedFloat<f64>,
        event: Event,
    }

    #[rustfmt::skip]
    #[derive(Debug, Default)]
    struct Stats {
        /// How many times system was in state S
        counts:	        FnvHashMap<State, u32>,

        /// How many times system dropped client in state S
        drops:	        FnvHashMap<State, u32>,

        /// Time spent in state S
        t_state:        FnvHashMap<State, f64>,

        /// How long system was in serving state (From First to SecondFinished)
        served_time:    f64,

        /// How many clients were served
        served_clients: u32,

        /// How many clients arrived including dropped
        arrived:        u32,
    }

    /// Map state to number of clients in that state
    fn client_number(state: State) -> usize {
        match state {
            State::Empty => 0,
            State::First | State::Second => 1,
            _ => 2,
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

    /// Print normalized table-like report for all states
    fn report<T>(title: &str, counts: &FnvHashMap<State, T>)
    where
        T: Copy + Into<f64>,
    {
        println!("{}", title);
        let events: f64 = counts.values().copied().map_into::<f64>().sum();
        let states: Vec<_> = counts.keys().copied().sorted().collect();

        for state in states {
            println!(
                "{:?}:   \t{}",
                state,
                Into::<f64>::into(counts[&state]) / events
            );
        }

        println!();
    }

    pub struct Simulation<T: Distribution<f64>> {
        stats: Stats,
        window: BinaryHeap<Reverse<Pair>>,
        iterations: u64,
        distributions: FnvHashMap<Event, T>,
        log_tail: u64,
    }

    impl<T> Simulation<T>
    where
        T: Distribution<f64>,
    {
        pub fn new(distributions: FnvHashMap<Event, T>, iterations: u64) -> Simulation<T> {
            Simulation {
                stats: Stats::default(),
                window: BinaryHeap::new(),
                iterations: iterations,
                distributions: distributions,
                log_tail: 0,
            }
        }

        pub fn set_tail(&mut self, new_tail: u64) {
            self.log_tail = new_tail;
        }

        /// Report of collected stats
        pub fn print_report(&self) {
            let dropful_counts: FnvHashMap<_, _> = self
                .stats
                .counts
                .iter()
                .map(|(&state, count)| (state, count + *self.stats.drops.get(&state).unwrap_or(&0)))
                .collect();

            // How long there was {0, 1, 2} clients in the system
            let mut t_client = [0f64; 3];
            for (&i, time) in self.stats.t_state.iter() {
                t_client[client_number(i)] += time
            }

            let dropped: u32 = self.stats.drops.values().sum();

            report("\nTime in states: ", &self.stats.t_state);
            report("Entries in states: ", &self.stats.counts);
            report("Entries in states with dropouts: ", &dropful_counts);

            println!(
                "Dropout:                   {dropout}\n\
                 Average serving time:      {time}\n\
                 Average number of clients: {number}",
                dropout = (dropped as f64) / (self.stats.arrived as f64),
                time = self.stats.served_time / (self.stats.served_clients as f64),
                number = (t_client[1] + 2.0f64 * t_client[2]) / t_client.iter().sum::<f64>()
            );
        }

        pub fn simulate(&mut self, prng: &mut StdRng) -> Result<(), &str> {
            // generate dt and insert (from_time + dt, event) in a window
            macro_rules! pusher {
                ($t:expr, $event:expr) => {{
                    let dt: f64 = self.distributions[&$event].sample(prng).into();
                    self.window.push(Reverse(Pair {
                        time: ($t + dt).into(),
                        event: $event,
                    }));
                }};
            }

            // time of last change of state
            let mut prev = 0f64;
            let mut state = State::Empty;
            // basically two floats on stack
            let mut arriving_times = Queue::<f64>::default();

            self.window.push(Reverse(Pair {
                time: 0.0.into(),
                event: Event::Arrived,
            }));

            // get current event, resubscribe it if needed, determine new state,
            // generate new events if we got a state and not a transition
            // collect statistics everywhere
            for i in 0..self.iterations {
                let current = self.window.pop().unwrap().0;
                if self.iterations - i < self.log_tail {
                    println!(
                        "{}: [{:?}] {:?} ==> [{:?}]",
                        current.time.0,
                        state,
                        current.event,
                        advance(state, current.event)?
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

                let new_state = advance(state, current.event)?;
                match new_state {
                    Right(Transition::Dropping) => {
                        self.stats
                            .drops
                            .entry(state)
                            .and_modify(|cnt| *cnt += 1)
                            .or_default();
                        continue;
                    }
                    Left(_) if current.event == Event::Arrived => {
                        arriving_times.push_back(current.time.0);
                        pusher!(current.time.0, Event::FirstFinished);
                    }
                    Left(State::Second) => pusher!(current.time.0, Event::SecondFinished),
                    _ => (),
                }
                self.stats
                    .t_state
                    .entry(state)
                    .and_modify(|time| *time += current.time.0 - prev)
                    .or_default();

                prev = current.time.0;
                state = new_state.left().unwrap();

                self.stats
                    .counts
                    .entry(state)
                    .and_modify(|cnt| *cnt += 1)
                    .or_default();
            }

            Ok(())
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

    let mut distributions = FnvHashMap::default();

    distributions.insert(Event::Arrived, Exp::new(args.lambda).unwrap());
    distributions.insert(Event::FirstFinished, Exp::new(args.mu1).unwrap());
    distributions.insert(Event::SecondFinished, Exp::new(args.mu2).unwrap());

    let mut simulation = Simulation::new(distributions, args.iterations * 1_000_000);

    let seed = args.seed.unwrap_or(rand::thread_rng().gen());
    simulation.set_tail(args.tail);

    let mut prng: StdRng = SeedableRng::seed_from_u64(seed);

    if let Err(error) = simulation.simulate(&mut prng) {
        panic!("Error: {}, seed: {}", error, seed);
    }

    simulation.print_report();
}
