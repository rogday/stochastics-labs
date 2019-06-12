#include <map>
#include <vector>
#include <random>
#include <iostream>

int main(int argc, char *argv[]) {
  using distribution_t = std::exponential_distribution<double>;

  std::string events[3] = {"client arrived", "first finished",
                           "second finished"};
  std::string states[7] = {"EMPTY", "FIRST", "SECOND", "WAITING",
                           "BOTH",  "DROP",  "INVALID"};

  enum event_t { ARRIVED = 0, FIRST_FINISHED, SECOND_FINISHED };
  enum state_t { EMPTY = 0, FIRST, SECOND, WAITING, BOTH, DROP, INVALID };

  if (argc < 5) {
    std::cerr << "not enough arguments!\nlambda, m1, m2, max_time";
    return -1;
  }

  //Марк, не бей, автоформат всё испортил, но ты видел, что я пытался

  //                 EMPTY     FIRST    SECOND   WAITING  BOTH
  std::vector<std::vector<state_t>> event_to_state = {
      /* ARRIVED */ {FIRST, DROP, BOTH, DROP, DROP},
      /* FIRST_FINISHED */ {INVALID, SECOND, INVALID, INVALID, WAITING},
      /* SECOND_FINISHED */ {INVALID, INVALID, EMPTY, SECOND, SECOND},
  };

  double lambda = atof(argv[1]);
  double m1 = atof(argv[2]);
  double m2 = atof(argv[3]);
  double time_max = atof(argv[4]);

  std::mt19937_64 generator(std::random_device{}());

  struct stat_t {
    std::size_t clients = 0;
    std::size_t drop = 0;
    double average_workload = 0;
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

  state_t state = EMPTY;
  for (auto [time, event] : timeline) {
    if (argc > 5) {
      std::cout << "[PROCESSING]: " << time << " " << events[event]
                << std::endl;
      std::cout << "[INFO]: " << states[state] << std::endl;
    }

    if (event == ARRIVED)
      ++stats.clients;

    state_t new_state = event_to_state[event][state];

    switch (new_state) {
    case INVALID:
      break;

    case DROP:
      ++stats.drop;
      break;

    default:
      state = new_state;
      break;
    }
  }

  std::cout << "dropout: " << stats.drop / double(stats.clients) << std::endl;
}