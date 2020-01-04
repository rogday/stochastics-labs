use enum_map::Enum;

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
pub enum Transition {
    Dropping,
}

#[derive(Debug)]
pub enum SimulationError {
    InvalidState,
}
