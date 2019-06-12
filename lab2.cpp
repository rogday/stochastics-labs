#include <queue>
#include <random>
#include <iostream>
#include <thread>
#include <chrono>
/**
 * Есть два стула. На какой сам сядешь, на какой мать посадишь.
 * M1, M2 - интенсивность, p - приход посетителя
 */

namespace hardbass {

class simulation_t {
  using distribution_t = std::exponential_distribution<double>;
  static constexpr auto ITERATIONS = 20'000;

public:
  simulation_t(double lambda, double m1, double m2)
      : client{1.0 / lambda}, first{1.0 / m1}, second{1.0 / m2} {}

  simulation_t(const simulation_t &) = delete;
  simulation_t(simulation_t &&) = delete;

  void run() {
    std::mt19937_64 gen(std::random_device{}());
    // std::uint64_t current_time = 0;

    /**
     * 1. Предположим, что оба стула пустые.
     * 2. Заходит клиент, сажаем его на первый.
     * 3. Добавляем к текущему времени время ожидания.
     *
     */
    bool first_occupied = false, second_occupied = false;
    double eta_first = 0.0, eta_second = 0.0, until_next_client = client(gen);

    std::size_t clients = 0, dropped = 0;

    for (std::size_t iteration = 0; iteration < ITERATIONS; ++iteration) {
      std::cout << "I: " << iteration
                << ", F: " << (first_occupied ? "busy" : "empty")
                << ", S:" << (second_occupied ? "busy" : "empty") << std::endl;

      std::cout << "dropout: " << double(dropped) / clients << " " << clients
                << " " << dropped << std::endl;

      std::cout << eta_first << " " << eta_second << " " << until_next_client
                << std::endl;

      // std::this_thread::sleep_for(std::chrono::milliseconds(300));

      if (until_next_client <= 0.0) { //клиент пришёл
        ++clients;
        if (!first_occupied) { //если стул свободен - займёт, иначе идёт нахуй
          eta_first = first(gen);
          first_occupied = true;
        } else
          ++dropped;

        until_next_client = client(gen);
      }

      if (eta_first <= 0 && first_occupied) {
        if (second_occupied)
          eta_first = eta_second;
        else {
          first_occupied = false, second_occupied = true;
          eta_second = second(gen);
        }
      } else
        --eta_first;

      if (eta_second <= 0 && second_occupied)
        second_occupied = false;
      else
        --eta_second;

      --until_next_client;
    }
  }

private:
  /** @note first -- стул с пиками, second -- второй стул */
  distribution_t client, first, second;
};
} // namespace hardbass

// lamda = 3, m1 = 20, m2 = 1
// dropout
int main(int argc, char *argv[]) {
  // hardbass::simulation_t simulation(10.0, 12.0, 10.0);
  hardbass::simulation_t simulation(1, 1, 1);

  simulation.run();

  return 0;
}