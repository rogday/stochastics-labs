/*use rand::{rngs::StdRng, Rng, SeedableRng};
use rand_distr::Exp;

use structopt::StructOpt;

use fnv::FnvHashMap;

mod shoeshine_shop {
    use std::cmp::Reverse;
    use std::collections::{BinaryHeap, VecDeque};

    use fnv::FnvHashMap;
    use ordered_float::*;

    use rand::rngs::StdRng;
    use rand_distr::Distribution;

    #[derive(Debug, Copy, Clone, PartialEq, Ord, Eq, PartialOrd, Hash)]
    #[repr(usize)]
    pub enum Event {
        Arrived = 0,
        FirstFinished,
        SecondFinished,
    }

    #[derive(Debug, Copy, Clone, PartialEq, Ord, Eq, PartialOrd, Hash)]
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
        //state
        counts:			FnvHashMap<State, u32>,
        drops:			FnvHashMap<State, u32>,

        //time
        t_state:			FnvHashMap<State, f64>,
        //there can be 0, 1 or 2 clients in the system
        t_client:			[f64;  3],

        served_time:            f64,
        served_clients:         u32,

        arrived:        u32,
    }

    fn client_number(state: State) -> usize {
        use State::*;

        match state {
            Empty => 0,
            First | Second => 1,
            _ => 2,
        }
    }

    fn advance(state: State, event: Event) -> State {
        use Event::*;
        use State::*;

        match event {
            Arrived => match state {
                Empty => First,
                Second => Both,
                _ => Dropping,
            },
            FirstFinished => match state {
                First => Second,
                Both => Waiting,
                _ => Invalid,
            },
            SecondFinished => match state {
                Second => Empty,
                Waiting => Second,
                Both => First,
                _ => Invalid,
            },
        }
    }

    fn report<T>(title: &str, counts: &FnvHashMap<State, T>)
    where
        T: Copy + Into<f64>,
    {
        println!("{}", title);
        let events: f64 = counts.values().copied().map(Into::<f64>::into).sum();
        let mut states: Vec<_> = counts.keys().copied().collect();
        states.sort();

        for state in states {
            println!(
                "{:?}: {}",
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

        pub fn print_report(&mut self) {
            let dropful_counts: FnvHashMap<_, _> = self
                .stats
                .counts
                .iter()
                .map(|(&state, count)| (state, count + *self.stats.drops.get(&state).unwrap_or(&0)))
                .collect();

            report("\ntime in states: ", &self.stats.t_state);
            report("entries in states: ", &self.stats.counts);
            report("entries in states with dropouts: ", &dropful_counts);

            let dropped: u32 = self.stats.drops.values().sum();

            println!(
                "dropout: {dropout}\naverage serving time: {time}\naverage number of clients: {number}",
                dropout = (dropped as f64) / (self.stats.arrived as f64),
                time = self.stats.served_time / (self.stats.served_clients as f64),
                number = (self.stats.t_client[1] + 2.0f64 * self.stats.t_client[2])
                    / self.stats.t_client.iter().sum::<f64>()
            );
        }

        pub fn simulate(&mut self, prng: &mut StdRng) -> bool {
            macro_rules! pusher {
                ($t:expr, $event:expr) => {{
                    let dt: f64 = self.distributions[&$event].sample(prng).into();
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
                event: Event::Arrived,
            }));

            for i in 0..self.iterations {
                let event = self.window.pop().unwrap().0;
                if self.iterations - i < self.log_tail {
                    println!(
                        "{}: [{:?}] {:?} ==> [{:?}]",
                        event.time.0,
                        state,
                        event.event,
                        advance(state, event.event)
                    );
                }
                match event.event {
                    Arrived => {
                        self.stats.arrived += 1;
                        pusher!(event.time.0, Arrived);
                    }
                    SecondFinished => {
                        self.stats.served_time += event.time.0 - arriving_times.front().unwrap();
                        arriving_times.pop_front();
                        self.stats.served_clients += 1;
                    }
                    _ => (),
                }

                use Event::*;
                use State::*;

                let new_state = advance(state, event.event);
                match new_state {
                    Invalid => return false,
                    Dropping => {
                        self.stats
                            .drops
                            .entry(state)
                            .and_modify(|cnt| *cnt += 1)
                            .or_default();
                        continue;
                    }
                    First | Both if event.event == Arrived => {
                        arriving_times.push_back(event.time.0);
                        pusher!(event.time.0, FirstFinished);
                    }
                    Second => pusher!(event.time.0, SecondFinished),
                    _ => (),
                }
                self.stats
                    .t_state
                    .entry(state)
                    .and_modify(|time| *time += event.time.0 - prev)
                    .or_default();

                self.stats.t_client[client_number(state)] += event.time.0 - prev;

                prev = event.time.0;
                state = new_state;

                self.stats
                    .counts
                    .entry(state)
                    .and_modify(|cnt| *cnt += 1)
                    .or_default();
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
    use shoeshine_shop::*;
    let args = Args::from_args();

    let mut dists = FnvHashMap::default();

    dists.insert(Event::Arrived, Exp::new(args.lambda).unwrap());
    dists.insert(Event::FirstFinished, Exp::new(args.mu1).unwrap());
    dists.insert(Event::SecondFinished, Exp::new(args.mu2).unwrap());

    let mut simulation = Simulation::new(dists, args.iterations * 1_000_000);

    let seed = args.seed.unwrap_or(rand::thread_rng().gen());
    simulation.set_tail(args.tail);

    let mut prng: StdRng = SeedableRng::seed_from_u64(seed);

    if !simulation.simulate(&mut prng) {
        panic!("Error: invalid state reached, seed: {}", seed);
    }

    simulation.print_report();
}
*/

macro_rules! make_enum{
    ($name:ident, $($element:ident)+) => {
        #[derive(Debug)]
        enum $name{
            $($element),+
        }
    }
}

macro_rules! advance_both{
    (
        ($first:ident $($first_n:ident)+),
        ($second:ident $($second_n:ident)+)
    ) => {
        $first => $second,
        advance_both!($($first_n)+, $($second_n)+)
    }
}

macro_rules! match_from_table {
    (
        $state:ident,
        $event:ident
        - $($first_row:ident)+,
        $($first_element:ident: $($element:ident)+),+
    ) => {
        make_enum![$state, $($first_row)+];
        make_enum![$event, $($first_element)+];

        fn advance(state: $state, event: $event) -> $state{
            match (event){
                $(
                    $event::$first_element => match(state){
                        $state::A => $state::B,
                       // $(
                        advance_both!($($element)+, $($element)+)
                            //$state::$first_row => $state::$element
                        //),+
                    }
                ),+
            }
        }

    }
}

match_from_table![State, Event
                 -  A B,
                 C: B B,
                 D: B B
                 ];

fn main() {
    let a = State::A;
    let b = Event::C;
    println!("{:?} {:?}", a, b);
}
