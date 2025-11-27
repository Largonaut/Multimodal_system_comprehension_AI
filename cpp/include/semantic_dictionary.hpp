/**
 * Semantic Dictionary - Memory-Mapped 15M Entry Hash Table
 *
 * High-performance lookup for English → GREMLIN code translation.
 * Memory-mapped for zero load time and OS-managed caching.
 *
 * Security: Local file only, no network, complete control.
 * Performance: O(1) lookup, ~1.1GB file, sub-microsecond access.
 *
 * Author: GREMLIN Team
 * License: MIT
 */

#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <optional>
#include <memory>
#include <vector>

namespace gremlin {

/**
 * Dictionary entry
 */
struct DictionaryEntry {
    std::string word;         // English word
    std::string gremlin_code; // 3-character GREMLIN code
    uint32_t frequency;       // Corpus frequency (optional)
};

/**
 * Semantic Dictionary
 *
 * Maps 15 million English words/phrases to GREMLIN codes.
 * Uses memory-mapped file for zero-copy access.
 */
class SemanticDictionary {
public:
    /**
     * Constructor - loads dictionary from file
     *
     * @param dictionary_path Path to dictionary JSON file
     */
    explicit SemanticDictionary(const std::string& dictionary_path);

    /**
     * Destructor - unmaps memory
     */
    ~SemanticDictionary();

    // Non-copyable (memory-mapped resource)
    SemanticDictionary(const SemanticDictionary&) = delete;
    SemanticDictionary& operator=(const SemanticDictionary&) = delete;

    // Movable
    SemanticDictionary(SemanticDictionary&&) noexcept;
    SemanticDictionary& operator=(SemanticDictionary&&) noexcept;

    /**
     * Lookup GREMLIN code for English word
     *
     * @param word English word/phrase
     * @return GREMLIN code, or nullopt if not found
     */
    [[nodiscard]] std::optional<std::string> lookup(std::string_view word) const;

    /**
     * Check if word exists in dictionary
     *
     * @param word English word/phrase
     * @return true if word exists
     */
    [[nodiscard]] bool contains(std::string_view word) const noexcept;

    /**
     * Get dictionary size
     *
     * @return Number of entries
     */
    [[nodiscard]] size_t size() const noexcept {
        return entry_count_;
    }

    /**
     * Get all entries (for debugging/export)
     *
     * @return Vector of dictionary entries
     */
    [[nodiscard]] std::vector<DictionaryEntry> get_all_entries() const;

    /**
     * Get statistics
     *
     * @return Statistics string
     */
    [[nodiscard]] std::string get_statistics() const;

private:
    // Memory-mapped file
    void* mmap_ptr_ = nullptr;
    size_t mmap_size_ = 0;
    int fd_ = -1;  // File descriptor (Unix) or HANDLE (Windows)

    // Hash table for fast lookup
    struct HashEntry {
        uint32_t hash;
        uint32_t word_offset;      // Offset in memory map
        uint32_t code_offset;      // Offset in memory map
        uint16_t word_length;
        uint16_t code_length;
    };

    std::vector<HashEntry> hash_table_;
    size_t entry_count_ = 0;

    /**
     * Load dictionary from JSON
     *
     * @param path Path to dictionary JSON
     */
    void load_from_json(const std::string& path);

    /**
     * Build memory-mapped binary file
     *
     * @param json_path Input JSON path
     * @param bin_path Output binary path
     */
    void build_binary(const std::string& json_path, const std::string& bin_path);

    /**
     * Load from memory-mapped binary file
     *
     * @param bin_path Binary file path
     */
    void load_binary(const std::string& bin_path);

    /**
     * Hash function (FNV-1a)
     *
     * @param str String to hash
     * @return 32-bit hash
     */
    static uint32_t hash_string(std::string_view str) noexcept;

    /**
     * Map file into memory
     *
     * @param path File path
     */
    void map_file(const std::string& path);

    /**
     * Unmap memory
     */
    void unmap();
};

/**
 * Binary file format for memory-mapped dictionary
 *
 * Header (64 bytes):
 *   - Magic: "GREMLIN1" (8 bytes)
 *   - Version: uint32_t
 *   - Entry count: uint64_t
 *   - Hash table size: uint64_t
 *   - Reserved: 36 bytes
 *
 * Hash Table (entry_count * 20 bytes):
 *   - Hash: uint32_t
 *   - Word offset: uint32_t
 *   - Code offset: uint32_t
 *   - Word length: uint16_t
 *   - Code length: uint16_t
 *   - Padding: 6 bytes
 *
 * String Data (variable):
 *   - All words and codes (null-terminated)
 */

} // namespace gremlin
