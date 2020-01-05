mod enums;
pub use enums::*;

use enum_map::EnumMap;
use itertools::*;

#[derive(Debug, Default)]
pub struct Stats {
    /// How many times the system was in the state S
    pub counts: EnumMap<State, u32>,

    /// How many times the system dropped client in the state S
    pub drops: EnumMap<State, u32>,

    /// Time spent in the state S
    pub t_state: EnumMap<State, f64>,

    /// How long the system was in serving state (From First to SecondFinished)
    pub served_time: f64,

    /// How many clients were served
    pub served_clients: u32,

    /// How many clients arrived including dropped
    pub arrived: u32,
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

impl From<Stats> for Report {
    fn from(stats: Stats) -> Self {
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
