#include "edmir/runtime/host_phase_timing.hpp"
#include <iostream>
#include <string>

using namespace edmir::runtime;

struct Backend {
  std::int64_t ticks = 0;
  bool fail_execute = false, fail_wait = false;
  std::string order;
  void execute() {
    order += 'E';
    if (fail_execute) throw std::runtime_error("execute failed");
    ticks += 7;
  }
  void synchronize() {
    order += 'S';
    if (fail_wait) throw std::runtime_error("wait failed");
    ticks += 11;
  }
};

void require(bool value) { if (!value) throw std::runtime_error("test failed"); }

int main() {
  Backend b;
  const auto clock = [&] { return std::chrono::steady_clock::time_point(std::chrono::nanoseconds(b.ticks)); };
  auto samples = measure_host_phases(b, 3, clock);
  require(b.order == "ESESES" && samples.size() == 3);
  for (const auto& s : samples) require(s.submit_ns == 7 && s.completion_wait_ns == 11 && s.wall_ns == 18);
  for (int bad : {-1, 0, 65}) {
    const auto before = b.order;
    bool rejected = false;
    try { (void)measure_host_phases(b, bad, clock); } catch (const std::invalid_argument&) { rejected = true; }
    require(rejected && b.order == before);
  }
  b.order.clear(); b.fail_execute = true;
  bool failed = false;
  try { (void)measure_host_phases(b, 2, clock); } catch (const std::runtime_error&) { failed = true; }
  require(failed && b.order == "E");
  b.order.clear(); b.fail_execute = false; b.fail_wait = true; failed = false;
  try { (void)measure_host_phases(b, 2, clock); } catch (const std::runtime_error&) { failed = true; }
  require(failed && b.order == "ES");
  b.fail_wait = false; failed = false; int counter = 0;
  auto backward = [&] { return std::chrono::steady_clock::time_point(std::chrono::nanoseconds(-++counter)); };
  try { (void)measure_host_phases(b, 1, backward); } catch (const std::runtime_error&) { failed = true; }
  require(failed);
  std::cout << "host phase timing: identity, order, limits, execute/wait failure, clock reversal passed\n";
}
