use enum_map::{enum_map, EnumMap};
use rand::{rngs::SmallRng, Rng, SeedableRng};
use rand_distr::Exp;

use shoeshine_shop::{statistics::{enums::*, Report}, Simulation};

const EPS: f64 = 0.01;

// release
#[cfg(not(debug_assertions))]
const ITERATIONS: u64 = 50_000_000;

// debug
#[cfg(debug_assertions)]
const ITERATIONS: u64 = 1_000_000;

fn run(lambda: f64, mu1: f64, mu2: f64) -> Report {
    let simulation = Simulation {
        iterations:    ITERATIONS,
        log_tail:      0,
        distributions: enum_map! {
            Event::Arrived        => Exp::new(lambda).unwrap(),
            Event::FirstFinished  => Exp::new(mu1).unwrap(),
            Event::SecondFinished => Exp::new(mu2).unwrap(),
        },
    };

    let seed = rand::thread_rng().gen();
    let mut prng: SmallRng = SeedableRng::seed_from_u64(seed);

    // if the test fails, it will be printed out
    println!("Using seed: {}", seed);

    simulation.simulate(&mut prng).expect("Simulation failed")
}

fn approx_eq(reference: f64, actual: f64, msg: &str) {
    assert!(
        (reference - actual).abs() < EPS,
        format!("{}: expected {}, got {}", msg, reference, actual)
    );
}

fn compare_maps(title: &str, reference: &EnumMap<State, f64>, actual: &EnumMap<State, f64>) {
    println!("Comparing \"{}\"...", title);
    for (state, &value) in reference.iter() {
        approx_eq(value, actual[state], &format!("Wrong value in {:?}", state));
    }
}

fn compare(reference: &Report, actual: &Report) {
    compare_maps("Time", &actual.t_states, &reference.t_states);
    compare_maps("Counts", &actual.counts, &reference.counts);
    compare_maps("Counts with drops", &actual.dropful_counts, &reference.dropful_counts);

    approx_eq(actual.dropout, reference.dropout, "Dropout is wrong");
    approx_eq(actual.t_serving_avg, reference.t_serving_avg, "Avg. serving time is wrong");
    approx_eq(actual.n_clients_avg, reference.n_clients_avg, "Avg. n of clients is wrong");
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

        dropout:       0.6963212748684511,
        t_serving_avg: 1.7641820719255652,
        n_clients_avg: 1.6072245422261517,
    };

    let actual = run(3.0, 20.0, 1.0);

    compare(&reference, &actual);
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

        dropout:       0.555669964286005,
        t_serving_avg: 2.250242037846493,
        n_clients_avg: 1.000111650931377,
    };

    let actual = run(1.0, 1.0, 1.0);

    compare(&reference, &actual);
}
