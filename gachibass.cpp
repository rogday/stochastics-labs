#include <iostream>
#include <iomanip>
#include <random>
#include <numeric>
#include <algorithm>
#include <array>
#include <queue>
#include <string>
#include <functional>

namespace shoeshine_shop {

enum event_t { ARRIVED, FIRST_FINISHED, SECOND_FINISHED };
enum state_t { EMPTY, FIRST, SECOND, WAITING, BOTH, DROP, INVALID };

using distribution_t = std::function<double()>;

using pair_t = struct {
  double time;
  event_t event;
};

struct stats_t {
  std::array<size_t, DROP> state_counts;
  std::array<size_t, DROP> state_counts_with_drop;

  std::array<double, DROP> time_in_state;
  std::array<double, 3> time_in_client;

  double served_time;

  size_t served_clients, arrived_clients, dropped_clients;
};

class simulation_t {

private:
  template <typename T>
  auto print_tables(std::string_view title, T const &counts) {
    std::cout << title << "\n";
    auto events = std::accumulate(std::begin(counts), std::end(counts), 0.0);
    for (size_t i = 0; i < DROP; ++i)
      std::cout << state_names[i] << ": " << counts[i] / double(events)
                << std::endl;
    std::cout << std::endl;
  }

  void debug_view(pair_t &event, state_t &state) {
    std::cout << "\t"
              << std::setprecision(std::numeric_limits<double>::digits10 + 1)
              << event.time << ": [" << state_names[state] << "] "
              << event_names[event.event] << " ==> ["
              << state_names[event_to_state[event.event][state]] << "]"
              << std::endl;
  }

public:
  simulation_t(distribution_t arrival, distribution_t first_serving,
               distribution_t second_serving, std::uint64_t iterations = 50)
      : window{queue_comparator}, iterations{1'000'000 * iterations},
        distributions{arrival, first_serving, second_serving}, log_tail{0} {}

  auto set_tail(std::uint64_t new_tail) noexcept { log_tail = new_tail; }

  bool simulate() {

    static auto pusher = [&](double t, event_t event) {
      double dt = distributions[event]();
      window.push({t + dt, event});
    };

    state_t state = EMPTY;
    double prev = 0.0;
    std::queue<double> arriving_times;

    window.push({0.0, ARRIVED});
    for (std::uint64_t i = 0; i < iterations; ++i) {
      auto event = window.top();
      window.pop();

      if (iterations - i < log_tail)
        debug_view(event, state);

      switch (event.event) {
      case ARRIVED:
        ++statistics.arrived_clients;
        pusher(event.time, ARRIVED);
        break;
      case SECOND_FINISHED:
        statistics.served_time += event.time - arriving_times.front();
        arriving_times.pop();
        ++statistics.served_clients;
      }

      state_t new_state = event_to_state[event.event][state];
      switch (new_state) {
      case INVALID:
        return false;

      case DROP:
        ++statistics.state_counts_with_drop[state];
        ++statistics.dropped_clients;
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

      statistics.time_in_state[state] += event.time - prev;
      statistics.time_in_client[state_to_clients[state]] += event.time - prev;
      prev = event.time;

      state = new_state;
      ++statistics.state_counts[state];
    }
    return true;
  }

  void print_report() {
    std::transform(std::begin(statistics.state_counts),
                   std::end(statistics.state_counts),
                   std::begin(statistics.state_counts_with_drop),
                   std::begin(statistics.state_counts_with_drop),
                   std::plus<std::size_t>());

    print_tables("time in states: ", statistics.time_in_state);
    print_tables("entries in states: ", statistics.state_counts);
    print_tables("entries in states with dropouts: ",
                 statistics.state_counts_with_drop);

    std::cout << "dropout: "
              << statistics.dropped_clients / double(statistics.arrived_clients)
              << std::endl;

    std::cout << "average serving time: "
              << statistics.served_time / double(statistics.served_clients)
              << std::endl;

    std::cout << "average number of clients: "
              << (statistics.time_in_client[1] +
                  2 * statistics.time_in_client[2]) /
                     std::accumulate(std::begin(statistics.time_in_client),
                                     std::end(statistics.time_in_client), 0.0)
              << std::endl;
  }

private:
  static constexpr std::array<const char *, 3> event_names{
      {"ARRIVED", "FIRST_FINISHED", "SECOND_FINISHED"}};
  static constexpr std::array<const char *, 7> state_names{
      {"EMPTY", "FIRST", "SECOND", "WAITING", "BOTH", "DROP", "INVALID"}};
  static constexpr std::array<std::array<state_t, 5>, 3> event_to_state{
      {{FIRST, DROP, BOTH, DROP, DROP},
       {INVALID, SECOND, INVALID, INVALID, WAITING},
       {INVALID, INVALID, EMPTY, SECOND, FIRST}}};

  static constexpr std::array<size_t, DROP> state_to_clients{0, 1, 1, 2, 2};

  inline static const auto queue_comparator = [](pair_t const &left,
                                                 pair_t const &right) {
    return (left.time > right.time);
  };

  stats_t statistics{};
  std::priority_queue<pair_t, std::vector<pair_t>, decltype(queue_comparator)>
      window;
  std::uint64_t iterations;
  std::array<distribution_t, 3> distributions;
  std::uint64_t log_tail;
};
} // namespace shoeshine_shop

int main(int argc, char **argv) {
  if (argc < 5) {
    std::cerr
        << "not enough arguments!\nlambda, m1, m2, millions of iterations";
    return EXIT_FAILURE;
  }

  std::uint32_t seed = std::random_device{}();
  std::mt19937 gen(seed);

  std::exponential_distribution arrival(std::atof(argv[1]));
  std::exponential_distribution first_serving(std::atof(argv[2]));
  std::exponential_distribution second_serving(std::atof(argv[3]));

  shoeshine_shop::simulation_t simul(std::bind(arrival, std::ref(gen)),
                                     std::bind(first_serving, std::ref(gen)),
                                     std::bind(second_serving, std::ref(gen)),
                                     std::atoll(argv[4]));

  if (argc == 6) {
    seed = std::atol(argv[5]);
    gen.seed(seed);
    simul.set_tail(100);
  }

  if (!simul.simulate()) {
    std::cerr << "ERROR: INVALID STATE REACHED, SEED: " << seed << std::endl;
    return EXIT_FAILURE;
  }

  simul.print_report();
}
