use either::*;
use enum_map::EnumMap;
use ordered_float::*;
use rand::rngs::SmallRng;
use rand_distr::Distribution;

mod statistics;
mod utils;
pub use statistics::*;
use utils::*;

pub struct Simulation<T> {
    pub distributions: EnumMap<Event, T>,
    pub iterations:    u64,
    pub log_tail:      u64,
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
    pub fn simulate(&self, prng: &mut SmallRng) -> Result<Report, SimulationError> {
        let mut window = PriorityQueue3::new();

        // generate dt and insert (from_time + dt, event) in a window
        macro_rules! pusher {
            ($t:expr, $event:expr) => {{
                let dt: f64 = self.distributions[$event].sample(prng).into();

                window.push((($t + dt).into(), $event));
            }};
        }

        // various metrics
        let mut stats = Stats::default();
        let mut arrived_time = Queue::default();
        let mut last_state_change: f64 = 0.;

        // starting conditions
        let mut state = State::Empty;
        window.push((0.0.into(), Event::Arrived));

        // get current event, resubscribe it if needed, determine new state,
        // generate new events if we got a state and not a transition
        // collect statistics everywhere
        for i in 0..self.iterations {
            let (OrderedFloat(current_time), event) = window.pop();
            let new_state = advance(state, event)?;

            // TODO: check if this makes 2 loops - one from 0 to iterations-log_tail, and other with the rest
            if self.iterations - i < self.log_tail + 1 {
                println!("{:.10}: [{:?}] {:?} ==> [{:?}]", i, state, event, new_state);
            }

            match event {
                Event::Arrived => {
                    stats.arrived += 1;
                    pusher!(current_time, Event::Arrived);
                }
                Event::FirstFinished => (),
                Event::SecondFinished => {
                    stats.served_time += current_time - arrived_time.pop_front();
                    stats.served_clients += 1;
                }
            }

            match new_state {
                // transitioned to pseudostate
                Right(Transition::Dropping) => {
                    stats.drops[state] += 1;
                }
                // NOTE: shadowing occurs here
                // transitioned to real state
                Left(new_state) => {
                    if let Event::Arrived = event {
                        arrived_time.push_back(current_time);
                        pusher!(current_time, Event::FirstFinished);
                    } else if let State::Second = new_state {
                        pusher!(current_time, Event::SecondFinished)
                    }

                    stats.t_state[state] += current_time - last_state_change;
                    last_state_change = current_time;
                    state = new_state;
                    stats.counts[state] += 1;
                }
            }
        }

        // Transform collected data into Report with useful statistics
        Ok(Report::from(stats))
    }
}
