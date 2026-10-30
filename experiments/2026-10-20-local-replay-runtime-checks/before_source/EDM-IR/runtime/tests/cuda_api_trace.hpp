#pragma once
#include <string>
#include <vector>

// CPU test doubles only. No tensor arithmetic, GPU timings, or asynchronous
// device behavior are emulated. Handles are deterministic opaque identifiers.
namespace edmir::runtime::test {
void reset_api_trace();
void clear_api_trace();
std::vector<std::string> api_trace();
}
