use enum_map::Enum;

// used in priority queue hence Ord
#[derive(Enum, Debug, Copy, Clone, PartialOrd, PartialEq, Eq, Ord)]
#[repr(usize)]
pub enum Event {
    Arrived,
    FirstFinished,
    SecondFinished,
}

impl Default for Event {
    fn default() -> Self {
        // because if it leaks from custom queue, it will trigger simulation error
        Event::SecondFinished
    }
}

#[derive(Enum, Debug, Copy, Clone)]
#[repr(usize)]
pub enum State {
    Empty,
    First,
    Second,
    Waiting,
    Both,
}
#[derive(Debug)]
pub enum Transition {
    Dropping,
}

#[derive(Debug)]
pub enum SimulationError {
    InvalidState,
}
