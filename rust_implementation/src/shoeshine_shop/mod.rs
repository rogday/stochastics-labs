use enum_map::EnumMap;

use either::*;

use ordered_float::*;

use rand::rngs::SmallRng;
use rand_distr::Distribution;

mod utils;
use utils::*;

mod statistics;
pub use statistics::*;

#[derive(Default, Copy, Clone, Debug, Ord, Eq, PartialEq, PartialOrd)]
struct Pair {
    time: OrderedFloat<f64>,
    event: Event,
}

pub struct Simulation<T: Distribution<f64>> {
    iterations: u64,
    distributions: EnumMap<Event, T>,
    log_tail: u64,
}

/// Determine new state or pseudostate(Transition) from current state and incoming event
fn advance(state: State, event: Event) -> Result<Either<State, Transition>, SimulationError> {
    use State::*;

    // explicit matching to ensure compile time error in case of newly added state
    match event {
        Event::Arrived => Ok(match state {
            Empty => Left(First),
            Second => Left(Both),
            // first chair is occupied
            First | Waiting | Both => Right(Transition::Dropping),
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

        // various metrics
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
            // current time and event
            let Pair {
                time: OrderedFloat(time),
                event,
            } = window.pop();

            let new_state = advance(state, event)?;

            // TODO: check if this makes 2 loops - one from 0 to iterations-log_tail, and other with the rest
            if self.iterations - i < self.log_tail + 1 {
                println!("{:.10}: [{:?}] {:?} ==> [{:?}]", i, state, event, new_state);
            }

            match event {
                Event::Arrived => {
                    stats.arrived += 1;
                    pusher!(time, Event::Arrived);
                }
                Event::SecondFinished => {
                    stats.served_time += time - arriving_times.pop_front();
                    stats.served_clients += 1;
                }
                _ => (),
            }

            match new_state {
                Right(Transition::Dropping) => {
                    stats.drops[state] += 1;
                    continue;
                }
                // Left(_) = new state is not pseudostate
                Left(_) if event == Event::Arrived => {
                    arriving_times.push_back(time);
                    pusher!(time, Event::FirstFinished);
                }
                Left(State::Second) => pusher!(time, Event::SecondFinished),
                _ => (),
            }
            stats.t_state[state] += time - prev;

            prev = time;
            // dealt with Right in match above using continue
            state = new_state.left().unwrap();

            stats.counts[state] += 1;
        }

        // Transform collected data into Report with useful statistics
        Ok(Report::from(&stats))
    }
}
