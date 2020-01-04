use enum_map::{Enum, EnumMap};

use either::*;

use itertools::*;
use ordered_float::*;

use rand::rngs::SmallRng;
use rand_distr::Distribution;

mod utils;
use utils::*;

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

#[derive(Debug, Default)]
struct Stats {
    /// How many times the system was in the state S
    counts: EnumMap<State, u32>,

    /// How many times the system dropped client in the state S
    drops: EnumMap<State, u32>,

    /// Time spent in the state S
    t_state: EnumMap<State, f64>,

    /// How long the system was in serving state (From First to SecondFinished)
    served_time: f64,

    /// How many clients were served
    served_clients: u32,

    /// How many clients arrived including dropped
    arrived: u32,
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

/// Map state to number of clients in that state
// Not a hasher because otherwise Eq must be reimplemented as well
fn client_number(state: State) -> usize {
    match state {
        State::Empty => 0,
        State::First | State::Second => 1,
        State::Waiting | State::Both => 2,
    }
}

impl From<&Stats> for Report {
    fn from(stats: &Stats) -> Self {
        let mut dropful_counts = EnumMap::<State, u32>::new();

        for (state, count) in stats.counts.iter() {
            dropful_counts[state] = count + stats.drops[state];
        }

        // How long there was {0, 1, 2} clients in the system
        let mut t_client = [0f64; 3];
        for (state, time) in stats.t_state.iter() {
            t_client[client_number(state)] += time
        }

        let dropped: u32 = stats.drops.values().sum();

        Report {
            t_states: normalized(&stats.t_state),
            counts: normalized(&stats.counts),
            dropful_counts: normalized(&dropful_counts),

            dropout: (dropped as f64) / (stats.arrived as f64),
            t_serving_avg: stats.served_time / (stats.served_clients as f64),
            n_clients_avg: (t_client[1] + 2.0f64 * t_client[2]) / t_client.iter().sum::<f64>(),
        }
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

pub struct Simulation<T: Distribution<f64>> {
    iterations: u64,
    distributions: EnumMap<Event, T>,
    log_tail: u64,
}

impl<T> Simulation<T>
where
    T: Distribution<f64>,
{
    pub fn new(distributions: EnumMap<Event, T>, iterations: u64, log_tail: u64) -> Simulation<T> {
        // no assertions on distributions needed because EnumMap elements are stored in array and therefore always initialized
        Simulation {
            iterations,
            distributions,
            log_tail,
        }
    }

    pub fn simulate(&self, prng: &mut SmallRng) -> Result<Report, SimulationError> {
        let mut window = TreeMin3::new();

        // generate dt and insert (from_time + dt, event) in a window
        macro_rules! pusher {
            ($t:expr, $event:expr) => {{
                let dt: f64 = self.distributions[$event].sample(prng).into();

                window.push(Pair {
                    time: ($t + dt).into(),
                    event: $event,
                });
            }};
        }

        let mut stats = Stats::default();

        // time of last change of state
        let mut prev = 0f64;
        let mut state = State::Empty;
        // basically two floats on stack
        let mut arriving_times = Queue::<f64>::default();

        window.push(Pair {
            time: 0.0.into(),
            event: Event::Arrived,
        });

        // get current event, resubscribe it if needed, determine new state,
        // generate new events if we got a state and not a transition
        // collect statistics everywhere
        for i in 0..self.iterations {
            let current = window.pop();
            let new_state = advance(state, current.event)?;

            // TODO: check if this makes 2 loops - one from 0 to iterations-log_tail, and other with the rest
            if self.iterations - i < self.log_tail + 1 {
                println!(
                    "{:.10}: [{:?}] {:?} ==> [{:?}]",
                    i, state, current.event, new_state
                );
            }

            match current.event {
                Event::Arrived => {
                    stats.arrived += 1;
                    pusher!(current.time.0, Event::Arrived);
                }
                Event::SecondFinished => {
                    stats.served_time += current.time.0 - arriving_times.pop_front();
                    stats.served_clients += 1;
                }
                _ => (),
            }

            match new_state {
                Right(Transition::Dropping) => {
                    stats.drops[state] += 1;
                    continue;
                }
                Left(_) if current.event == Event::Arrived => {
                    arriving_times.push_back(current.time.0);
                    pusher!(current.time.0, Event::FirstFinished);
                }
                Left(State::Second) => pusher!(current.time.0, Event::SecondFinished),
                _ => (),
            }
            stats.t_state[state] += current.time.0 - prev;

            prev = current.time.0;
            state = new_state.left().unwrap();

            stats.counts[state] += 1;
        }

        // Transform collected data into Report with useful statistics
        Ok((&stats).into())
    }
}
