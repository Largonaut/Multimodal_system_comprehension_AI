# GREMLIN C++ - High-Performance Inference Engine

**Status:** Foundation complete, implementation in progress
**Performance Target:** 5-10x faster than Python
**Philosophy:** Built from scratch, machine precision, complete control

---

## Overview

High-performance C++ implementation of GREMLIN semantic compression engine. Designed to run tight on everything from 8-year-old iPads to AI servers.

**What We're Building:**
- Lightning-fast dictionary lookup (memory-mapped, O(1))
- SIMD-optimized tokenizer (128K vocabulary)
- Character-level fallback (100% coverage guarantee)
- Model inference integration (Gemma.cpp or custom)
- Secure socket layer (local IPC → P2P → Tor)

**Security:** Zero hub dependencies, no network calls, complete local control.

---

## Project Structure

```
cpp/
├── include/                    # Public headers
│   ├── gremlin_engine.hpp     # Main API
│   ├── gremlin_tokenizer.hpp  # 128K BPE tokenizer
│   ├── semantic_dictionary.hpp # 15M entry dictionary
│   └── trap_door.hpp          # Character fallback
│
├── src/                        # Implementation
│   ├── gremlin_engine.cpp
│   ├── gremlin_tokenizer.cpp
│   ├── semantic_dictionary.cpp
│   ├── trap_door.cpp
│   ├── main.cpp               # CLI tool
│   └── gremlin_dataloader.cpp # PyTorch extension
│
├── tests/                      # Test suite
│   ├── test_tokenizer.cpp
│   ├── test_dictionary.cpp
│   ├── test_trap_door.cpp
│   └── test_engine.cpp
│
├── examples/                   # Usage examples
│   ├── hello_world.cpp
│   ├── batch_translation.cpp
│   └── streaming.cpp
│
├── docs/                       # Documentation
│   ├── architecture.md
│   ├── performance.md
│   └── gemma_cpp_integration.md
│
└── CMakeLists.txt             # Build system
```

---

## Components

### 1. GremlinEngine (Main API)

**Header:** `include/gremlin_engine.hpp`
**Implementation:** `src/gremlin_engine.cpp`

**Purpose:** Main API for English ↔ GREMLIN translation

**Features:**
- Dictionary-based translation
- Model-based translation (optional)
- Hybrid mode (dictionary first, model fallback)
- Streaming API for large texts
- Comprehensive statistics

**Usage:**
```cpp
#include <gremlin_engine.hpp>

int main() {
    // Create engine
    gremlin::GremlinEngine engine(
        "models/tokenizer/gremlin_tokenizer.json",
        "models/dictionary/gremlin_dictionary.bin"
    );

    // Translate
    std::string gremlin = engine.translate("Hello world");
    std::string english = engine.decompress(gremlin);

    // Get stats
    auto stats = engine.get_stats();
    std::cout << "Compression: " << stats.compression_ratio << std::endl;
}
```

---

### 2. GremlinTokenizer (128K BPE)

**Header:** `include/gremlin_tokenizer.hpp`
**Implementation:** `src/gremlin_tokenizer.cpp`

**Purpose:** Byte-Pair Encoding tokenizer for GREMLIN

**Features:**
- 128K vocabulary (vs 256K Gemma base)
- SIMD-optimized UTF-8 validation
- Zero-copy where possible
- Batch operations
- Thread-safe

**Performance:**
- Encoding: 5-10x faster than Python
- Decoding: 3-5x faster than Python
- Memory: O(1) overhead

**Usage:**
```cpp
#include <gremlin_tokenizer.hpp>

gremlin::GremlinTokenizer tokenizer("gremlin_tokenizer.json");

// Encode
auto tokens = tokenizer.encode("Hello world", true);

// Decode
auto text = tokenizer.decode(tokens, true);

// Batch
std::vector<std::string> texts = {"Hello", "World"};
auto batch_tokens = tokenizer.encode_batch(texts);
```

---

### 3. SemanticDictionary (15M Entries)

**Header:** `include/semantic_dictionary.hpp`
**Implementation:** `src/semantic_dictionary.cpp`

**Purpose:** O(1) lookup for English → GREMLIN codes

**Features:**
- Memory-mapped file (1.1GB, zero load time)
- FNV-1a hashing (fast, good distribution)
- Robin Hood probing (excellent cache locality)
- OS-managed caching (shared across processes)

**Performance:**
- Lookup: <1 microsecond
- Load time: ~0ms (memory-mapped)
- Memory: 1.1GB (shared)

**File Format:**
```
Binary Format (.bin):
[Header: 64 bytes]
  - Magic: "GREMLIN1"
  - Version: uint32_t
  - Entry count: uint64_t

[Hash Table: N × 20 bytes]
  - Hash: uint32_t
  - Offsets: 2 × uint32_t
  - Lengths: 2 × uint16_t

[String Data: Variable]
  - Null-terminated strings
```

**Usage:**
```cpp
#include <semantic_dictionary.hpp>

gremlin::SemanticDictionary dict("gremlin_dictionary.bin");

// Lookup
auto code = dict.lookup("artificial intelligence");
if (code) {
    std::cout << "Code: " << *code << std::endl;  // "Ωµ§"
}

// Check existence
bool exists = dict.contains("hello");

// Statistics
std::cout << "Size: " << dict.size() << std::endl;  // 15,000,000
```

---

### 4. TrapDoor (Character Fallback)

**Header:** `include/trap_door.hpp`
**Implementation:** `src/trap_door.cpp`

**Purpose:** Character-level encoding for unknown words

**Features:**
- ASCII mapping (0-127)
- Extended Unicode mapping
- Pre-computed lookup tables
- Reversible encoding

**Guarantee:** 100% coverage (zero English leakage)

**Usage:**
```cpp
#include <trap_door.hpp>

gremlin::TrapDoor trap_door;

// Encode unknown word
auto encoded = trap_door.encode_word("Skibidi");  // Character-by-character

// Decode
auto decoded = trap_door.decode_word(encoded);  // "Skibidi"

// Check if trap door encoding
bool is_trap = trap_door.is_trap_door_code(encoded);
```

---

## Build System

### Requirements

**Compiler:**
- GCC 11+ (C++20 support)
- Clang 14+ (C++20 support)
- MSVC 19.30+ (Visual Studio 2022)

**Dependencies:**
- CMake 3.18+
- C++20 standard library

**Optional:**
- PyTorch C++ API (for data loader extension)
- pybind11 (for Python bindings)
- Gemma.cpp (for model inference)
- Google Test (for testing)

### Building

```bash
# Clone or navigate to cpp directory
cd F:/dev/GREMLIN_Claude_Code_Web_track/cpp

# Create build directory
mkdir build
cd build

# Configure
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTS=ON \
  -DBUILD_EXAMPLES=ON

# Build
cmake --build . --config Release -j8

# Test
ctest --output-on-failure

# Install (optional)
cmake --install . --prefix /usr/local
```

### CMake Options

```cmake
-DBUILD_TESTS=ON              # Build test suite
-DBUILD_EXAMPLES=ON           # Build examples
-DUSE_GEMMA_CPP=OFF           # Use Gemma.cpp for inference
-DBUILD_PYTHON_BINDINGS=ON    # Build PyTorch extension
-DENABLE_PROFILING=OFF        # Enable performance profiling
```

---

## Performance Targets

### Inference Speed

**Dictionary Operations:**
- Lookup: <1 microsecond
- 5-10x faster than Python

**Full Translation:**
- Typical sentence: <100ms
- 3-5x faster than Python

**Memory:**
- Dictionary: 1.1GB (memory-mapped, shared)
- Engine: <100MB overhead

### Training Acceleration (Data Loader)

**Python vs C++ Data Loader:**
- Tokenization: 5-10x faster
- JSONL parsing: 3-5x faster
- Batching: Zero-copy (no overhead)

**GPU Utilization:**
- Target: >90% (minimal data loading wait)

### Network Performance

**Local IPC (UNIX sockets):**
- Latency: <1ms
- Throughput: ~1GB/s

**P2P (Direct TCP + E2E encryption):**
- Latency: <50ms (local network)
- Throughput: ~100MB/s (limited by encryption)

**Tor Hidden Services:**
- Latency: <3 seconds (acceptable for async)
- Throughput: ~1-5MB/s (Tor bandwidth)

---

## Security Compliance

### Zero Hub Dependencies

**Excluded:**
- ❌ Hugging Face Transformers
- ❌ Hugging Face Hub
- ❌ Any library with phone-home code

**Included:**
- ✅ Standard C++ library
- ✅ Optional: libsodium (audited crypto)
- ✅ Optional: Gemma.cpp (Google, no hub)

### Security Principles

1. **No network calls** (except explicit sockets)
2. **No autoupdate mechanisms**
3. **No telemetry**
4. **No boolean toggles as security**
5. **Memory safety** (RAII, smart pointers, no raw pointers)

---

## Development Roadmap

### Phase 1: Core Infrastructure (Current)

- [x] Project structure
- [x] CMake build system
- [x] Header files (API design)
- [ ] Tokenizer implementation
- [ ] Dictionary implementation
- [ ] Trap Door implementation
- [ ] Engine implementation
- [ ] CLI tool

**Timeline:** 4-6 weeks

### Phase 2: Optimization

- [ ] SIMD optimizations (AVX2, NEON)
- [ ] Memory profiling
- [ ] Cache tuning
- [ ] Benchmark suite
- [ ] Performance documentation

**Timeline:** 2-3 weeks

### Phase 3: Model Integration

- [ ] Research Gemma.cpp compatibility
- [ ] Integrate or build model inference
- [ ] GGUF support
- [ ] LoRA weight loading
- [ ] Generation pipeline

**Timeline:** 3-4 weeks

### Phase 4: Python Bindings

- [ ] PyTorch C++ extension
- [ ] JSONL parser
- [ ] Thread pool
- [ ] Zero-copy batching
- [ ] Integration tests

**Timeline:** 2-3 weeks

### Phase 5: Socket Layer

- [ ] Socket abstraction
- [ ] Local IPC (UNIX sockets / Named pipes)
- [ ] Message format
- [ ] P2P sockets (libsodium E2E encryption)
- [ ] Tor integration

**Timeline:** 4-6 weeks

---

## Testing

### Unit Tests

**Framework:** Google Test

**Coverage:**
- Tokenizer (encode/decode, batch, edge cases)
- Dictionary (lookup, contains, statistics)
- Trap Door (encode/decode, Unicode)
- Engine (translate, decompress, streaming)

**Run:**
```bash
cd build
ctest --output-on-failure
```

### Integration Tests

**Scenarios:**
- End-to-end translation
- Streaming (large files)
- Batch processing
- Error handling

### Performance Benchmarks

**Metrics:**
- Tokens/second (encoding/decoding)
- Dictionary lookups/second
- Translation latency
- Memory usage

**Run:**
```bash
./gremlin-bench --iterations 10000
```

---

## Usage Examples

### Hello World

```cpp
#include <gremlin_engine.hpp>
#include <iostream>

int main() {
    auto engine = gremlin::create_engine(
        "models/tokenizer/gremlin_tokenizer.json",
        "models/dictionary/gremlin_dictionary.bin"
    );

    auto gremlin = engine.translate("Hello world");
    std::cout << "GREMLIN: " << gremlin << std::endl;

    auto english = engine.decompress(gremlin);
    std::cout << "English: " << english << std::endl;
}
```

### Batch Translation

```cpp
std::vector<std::string> texts = {
    "Artificial intelligence",
    "Machine learning",
    "Neural networks"
};

auto gremlin_texts = engine.translate_batch(texts);

for (size_t i = 0; i < texts.size(); ++i) {
    std::cout << texts[i] << " -> " << gremlin_texts[i] << std::endl;
}
```

### Streaming (Large Files)

```cpp
std::ifstream input("large_file.txt");
std::ofstream output("large_file.gremlin");

engine.translate_stream(input, output);

auto stats = engine.get_stats();
std::cout << "Compression ratio: " << stats.compression_ratio << std::endl;
```

---

## Contributing

**Development Philosophy:**
- Build from scratch where needed
- Learn from Gemma.cpp but don't be constrained
- Machine precision, no fatigue
- Complete control over security

**Code Style:**
- C++20 (concepts, ranges, std::span)
- Modern idioms (RAII, smart pointers)
- No raw pointers
- Comprehensive error handling

**Before Submitting:**
- Run clang-format
- Run clang-tidy
- All tests pass
- No memory leaks (valgrind)

---

## License

MIT License - See project root LICENSE file

---

## Credits

**Architecture:** David Berlekamp (The Architect)
**Implementation:** Claude Code (Co-Pilot)
**Philosophy:** "God Mode" - No compromises, frontier science
**Timeline:** November 2025 - Foundation built in collaboration

**Built on:**
- C++20 Standard
- Optional: Gemma.cpp (Google)
- Optional: libsodium (NaCl, audited crypto)

**Part of:** Project Guy - AI cognitive partner with permanent memory

---

**"C++ all the things. Machine precision. Complete control."**

*This is GREMLIN in C++. This is how we run tight.*
