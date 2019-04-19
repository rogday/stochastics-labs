#include <queue>
#include <random>
#include <iostream>
#include <thread>

/**
 * Имеется парикмахерская с посетителем и среднем раз в 12
 * минут (экспоненциальное распределение), 10 минут длится обычная стрижка, 15
 * - квалифицированная. 1 из 4 заказывает квалифицированную. Какие переменные и
 * случайные величины нужно ввести. Запрогать очередь, какие-то движения в ней
 */

using namespace std::chrono_literals;

class barbershop_t {
  enum haircut_t { ORDINARY, QUALIFIED };

  struct client_t {
    haircut_t haircut;
    double arrival_time;
  };

public:
  barbershop_t(std::int64_t client_delay, std::int64_t ordinary_duration,
               std::int64_t qualified_duration,
               double p) // p - ordinary prob
      : ordinary_duration(ordinary_duration),
        qualified_duration(qualified_duration), arrival(1.0 / client_delay),
        chose_haircut({p, 1 - p}) {}

  void run_simulation(std::uint64_t delay) {
    std::random_device rd;
    std::mt19937_64 gen(rd());

    std::int64_t current_time = 0;
    std::int64_t freeze_time = 0;

    std::int64_t sum = 0;

    while (true) {
      std::cout << "size: " << clients.size() << ", time: " << current_time
                << ", average: " << sum / double(current_time) << std::endl;

      std::this_thread::sleep_for(std::chrono::milliseconds(delay));

      if (clients.empty() ||
          current_time == (std::int64_t)clients.back().arrival_time) {
        double arrives = arrival(gen);

        haircut_t chosen_haircut = static_cast<haircut_t>(chose_haircut(gen));

        clients.push({chosen_haircut, current_time + arrives});
      }

      if (freeze_time-- <= 0 && !clients.empty() &&
          current_time >= (std::int64_t)clients.front().arrival_time) {
        const auto &client = clients.front();
        clients.pop();

        freeze_time = (client.haircut == haircut_t::ORDINARY)
                          ? ordinary_duration
                          : qualified_duration;
      }

      sum += clients.size();

      ++current_time;
    }
  }

private:
  std::int64_t ordinary_duration;
  std::int64_t qualified_duration;

  std::exponential_distribution<double> arrival;
  std::discrete_distribution<std::int64_t> chose_haircut;

  std::queue<client_t> clients;
};

int main(int argc, char *argv[]) {

  std::uint64_t delay = (argc == 2) ? atoll(argv[1]) : 0;

  barbershop_t barbershop(12, 10, 15, 0.75);
  barbershop.run_simulation(delay);

  return 0;
}