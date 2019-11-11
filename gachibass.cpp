#include <map>
#include <string>
#include <random>
#include <iostream>
#include <numeric>
#include <algorithm>
#include <queue>
#include <iomanip>

using distribution_t = std::exponential_distribution<double>;

enum event_t { ARRIVED = 0, FIRST_FINISHED, SECOND_FINISHED };
enum state_t { EMPTY = 0, FIRST, SECOND, WAITING, BOTH, DROP, INVALID };

const std::string event_names[3] = {"ARRIVED", "FIRST_FINISHED",
                                    "SECOND_FINISHED"};
const std::string state_names[7] = {"EMPTY", "FIRST", "SECOND", "WAITING",
                                    "BOTH",  "DROP",  "INVALID"};

const std::size_t state_to_clients[DROP] = {0, 1, 1, 2, 2};
// clang-format off
  //                         EMPTY    FIRST    SECOND   WAITING  BOTH
  const state_t event_to_state[3][5] = {
      /* ARRIVED */         {FIRST,   DROP,    BOTH,    DROP,    DROP},
      /* FIRST_FINISHED */  {INVALID, SECOND,  INVALID, INVALID, WAITING},
      /* SECOND_FINISHED */ {INVALID, INVALID, EMPTY,   SECOND,  FIRST},
  };
// clang-format on

struct stats_t {
  std::size_t state_counts[DROP]{}; // max feasible event - BOTH
  std::size_t state_counts_with_drop[DROP]{};

  double time_in_state[DROP]{};
  double time_in_client[3]{};

  double served_time = 0.0;
  std::size_t served_clients = 0;

  std::size_t arrived_clients = 0;
  std::size_t dropped_clients = 0;
};

struct pair_t {
  double time;
  event_t event;
};

template <typename T> void print_tables(std::string_view title, T &counts) {
  std::cout << title << std::endl;

  auto events = std::accumulate(counts, counts + DROP, 0.0);

  for (std::size_t i = 0; i < DROP; ++i)
    std::cout << state_names[i] << ": " << counts[i] / double(events)
              << std::endl;
  std::cout << std::endl;
};

void print_report(stats_t &stats) {
  std::transform(std::begin(stats.state_counts), std::end(stats.state_counts),
                 std::begin(stats.state_counts_with_drop),
                 std::begin(stats.state_counts_with_drop),
                 std::plus<std::size_t>());

  print_tables("time in states: ", stats.time_in_state);
  print_tables("entries in states: ", stats.state_counts);
  print_tables("entries in states with dropouts: ",
               stats.state_counts_with_drop);

  std::cout << "dropout: "
            << stats.dropped_clients / double(stats.arrived_clients)
            << std::endl;

  std::cout << "average serving time: "
            << stats.served_time / double(stats.served_clients) << std::endl;

  std::cout << "average number of clients: "
            << (stats.time_in_client[1] + 2 * stats.time_in_client[2]) /
                   std::accumulate(std::begin(stats.time_in_client),
                                   std::end(stats.time_in_client), 0.0)
            << std::endl;
}

void debug_view(pair_t &event, state_t &state) {
  std::cout << "\t"
            << std::setprecision(std::numeric_limits<double>::digits10 + 1)
            << event.time << ": [" << state_names[state] << "] "
            << event_names[event.event] << " ==> ["
            << state_names[event_to_state[event.event][state]] << "]"
            << std::endl;
}

int main(int argc, char *argv[]) {
  if (argc < 5) {
    std::cerr
        << "not enough arguments!\nlambda, m1, m2, millions of iterations";
    return EXIT_FAILURE;
  }

  double lambda = atof(argv[1]);
  double m1 = atof(argv[2]);
  double m2 = atof(argv[3]);
  std::uint64_t iterations = atoll(argv[4]) * 1'000'000;

  auto seed = std::random_device{}();
  std::mt19937_64 generator(seed);
  distribution_t dists[3] = {distribution_t(lambda), distribution_t(m1),
                             distribution_t(m2)}; // mean = 1/param

  auto cmp = [](const pair_t &left, const pair_t &right) {
    return (left.time > right.time);
  };
  std::priority_queue<pair_t, std::vector<pair_t>, decltype(cmp)> window(cmp);

  auto pusher = [&generator, &dists, &window](double t, event_t event) {
    double dt = dists[event](generator);
    window.push({t + dt, event});
  };

  state_t state = EMPTY;
  double prev = 0.0;
  std::queue<double> arriving_times;
  stats_t stats;

  window.push({0.0, ARRIVED});

  for (std::uint64_t i = 0; i < iterations; ++i) {
    auto event = window.top();
    window.pop();

    if (argc > 5)
      debug_view(event, state);

    switch (event.event) {
    case ARRIVED:
      ++stats.arrived_clients;
      pusher(event.time, ARRIVED);
      break;
    case SECOND_FINISHED:
      stats.served_time += event.time - arriving_times.front();
      arriving_times.pop();
      ++stats.served_clients;
    }

    state_t new_state = event_to_state[event.event][state];
    switch (new_state) {
    case INVALID:
      std::cerr << "ERROR: INVALID STATE REACHED, SEED: " << seed << std::endl;
      return EXIT_FAILURE;

    case DROP:
      ++stats.state_counts_with_drop[state];
      ++stats.dropped_clients;
      continue;

    case FIRST:
    case BOTH:
      if (event.event == ARRIVED) {
        arriving_times.push(event.time);
        pusher(event.time, FIRST_FINISHED);
      }
      break;

    case SECOND:
      pusher(event.time, SECOND_FINISHED);
      break;

    case EMPTY:
    case WAITING:
      break;
    }

    stats.time_in_state[state] += event.time - prev;
    stats.time_in_client[state_to_clients[state]] += event.time - prev;
    prev = event.time;

    state = new_state;
    ++stats.state_counts[state];
  }

  print_report(stats);

  return EXIT_SUCCESS;
  // arr=(10 10 10); for i in {0..2}; do for param in {1..100}; do
  // darr=("${arr[@]}"); darr[i]=${param}; echo "${darr[@]}" >> ../out.txt &&
  // ./lab2.exe ${darr[@]} 1000000 >> ../out.txt; done; done
}