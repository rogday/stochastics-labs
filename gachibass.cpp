#include <map>
#include <string>
#include <random>
#include <iostream>
#include <numeric>
#include <algorithm>
#include <queue>

int main(int argc, char *argv[]) {
  if (argc < 5) {
    std::cerr << "not enough arguments!\nlambda, m1, m2, max_time";
    return -1;
  }

  using distribution_t = std::exponential_distribution<double>;

  std::string event_names[3] = {"ARRIVED", "FIRST_FINISHED", "SECOND_FINISHED"};
  std::string state_names[7] = {"EMPTY", "FIRST", "SECOND", "WAITING",
                                "BOTH",  "DROP",  "INVALID"};

  enum event_t { ARRIVED = 0, FIRST_FINISHED, SECOND_FINISHED };
  enum state_t { EMPTY = 0, FIRST, SECOND, WAITING, BOTH, DROP, INVALID };

  std::size_t state_to_clients[DROP] = {0, 1, 1, 2, 2};

  // clang-format off
  //                         EMPTY    FIRST    SECOND   WAITING  BOTH
  state_t event_to_state[3][5] = {
      /* ARRIVED */         {FIRST,   DROP,    BOTH,    DROP,    DROP},
      /* FIRST_FINISHED */  {INVALID, SECOND,  INVALID, INVALID, WAITING},
      /* SECOND_FINISHED */ {INVALID, INVALID, EMPTY,   SECOND,  FIRST},
  };
  // clang-format on

  double lambda = atof(argv[1]);
  double m1 = atof(argv[2]);
  double m2 = atof(argv[3]);
  double time_max = atof(argv[4]);

  std::mt19937_64 generator(std::random_device{}());

  struct stats_t {
    std::size_t state_counts[DROP]{}; // max feasible event - BOTH
    std::size_t state_counts_with_drop[DROP]{};

    double time_in_state[DROP]{};
    double time_in_client[3]{}; // roflanEbalo

    double served_time = 0.0;
    std::size_t served_clients = 0;

    std::size_t arrived_clients = 0;
    std::size_t dropped_clients = 0;
  } stats;

  double times[3]{};
  distribution_t dists[3] = {distribution_t(lambda), distribution_t(m1),
                             distribution_t(m2)}; // mean = 1/param

  std::map<double, event_t> timeline;

  auto inserter = [&timeline, &generator](event_t event, double &t,
                                          distribution_t &dist) {
    double dt;
    do {
      dt = dist(generator);
    } while (!timeline.try_emplace(t + dt, event).second);
    t += dt;
  };

  for (std::size_t i = 0; i < 3; ++i)
    while (times[event_t(i)] < time_max)
      inserter(event_t(i), times[i], dists[i]);

  double prev = 0;
  state_t state = EMPTY;
  std::queue<double> arriving_times;

  for (auto [time, event] : timeline) {
    if (argc > 5) {
      std::cout << "[PROCESSING]: " << time << " " << event_names[event]
                << std::endl;
      std::cout << "[INFO]: " << state_names[state] << std::endl;
    }

    if (event == ARRIVED)
      ++stats.arrived_clients;

    state_t new_state = event_to_state[event][state];

    switch (new_state) {
    case INVALID:
      break;

    case DROP:
      ++stats.state_counts_with_drop[state];
      ++stats.dropped_clients;
      break;

    default:
      if (event == ARRIVED)
        arriving_times.push(time);
      else if (event == SECOND_FINISHED) {
        stats.served_time += time - arriving_times.front();
        arriving_times.pop();
        ++stats.served_clients;
      }

      stats.time_in_state[state] += time - prev;
      stats.time_in_client[state_to_clients[state]] += time - prev;
      prev = time;

      state = new_state;
      ++stats.state_counts[state];

      break;
    }
  }

  std::transform(std::begin(stats.state_counts), std::end(stats.state_counts),
                 std::begin(stats.state_counts_with_drop),
                 std::begin(stats.state_counts_with_drop),
                 std::plus<std::size_t>());

  auto report = [&state_names](std::string_view title, auto counts) {
    std::cout << title << std::endl;

    auto events = std::accumulate(counts, counts + DROP, 0.0);

    for (std::size_t i = 0; i < DROP; ++i)
      std::cout << state_names[i] << ": " << counts[i] / double(events)
                << std::endl;
    std::cout << std::endl;
  };

  report("time in states: ", stats.time_in_state);
  report("entries in states: ", stats.state_counts);
  report("entries in states with dropouts: ", stats.state_counts_with_drop);

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