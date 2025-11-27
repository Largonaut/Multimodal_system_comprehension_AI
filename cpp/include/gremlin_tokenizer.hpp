/**
 * GREMLIN Tokenizer - C++ Implementation
 *
 * High-performance tokenizer for GREMLIN's 128K vocabulary.
 * Zero dependencies, pure C++20, SIMD-optimized.
 *
 * Security: No network calls, no hub code, complete local control.
 * Performance: 5-10x faster than Python implementation.
 *
 * Author: GREMLIN Team
 * License: MIT
 */

#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>
#include <unordered_map>
#include <optional>
#include <span>

namespace gremlin {

/**
 * Special token IDs
 */
struct SpecialTokens {
    uint32_t bos_token_id = 0;      // <|endoftext|>
    uint32_t eos_token_id = 0;      // <|endoftext|>
    uint32_t pad_token_id = 1;      // <|padding|>
    uint32_t unk_token_id = 3;      // <|unk|>
    uint32_t gremlin_token_id = 2;  // <|gremlin|>
};

/**
 * BPE merge rule
 */
struct BPEMerge {
    std::string first;
    std::string second;
    std::string result;
    uint32_t priority;  // Lower = higher priority
};

/**
 * GREMLIN Tokenizer
 *
 * Implements Byte-Pair Encoding (BPE) with 128K vocabulary.
 * Optimized for 3-character GREMLIN codes.
 */
class GremlinTokenizer {
public:
    /**
     * Constructor - loads tokenizer from JSON file
     *
     * @param tokenizer_path Path to gremlin_tokenizer.json
     */
    explicit GremlinTokenizer(const std::string& tokenizer_path);

    /**
     * Encode text to token IDs
     *
     * @param text Input text
     * @param add_special_tokens Add BOS/EOS tokens
     * @return Vector of token IDs
     */
    std::vector<uint32_t> encode(
        std::string_view text,
        bool add_special_tokens = true
    ) const;

    /**
     * Encode multiple texts (batch)
     *
     * @param texts Vector of input texts
     * @param add_special_tokens Add BOS/EOS tokens
     * @param max_length Maximum sequence length (0 = no limit)
     * @return Vector of token ID vectors
     */
    std::vector<std::vector<uint32_t>> encode_batch(
        std::span<const std::string> texts,
        bool add_special_tokens = true,
        size_t max_length = 0
    ) const;

    /**
     * Decode token IDs to text
     *
     * @param token_ids Vector of token IDs
     * @param skip_special_tokens Skip special tokens in output
     * @return Decoded text
     */
    std::string decode(
        std::span<const uint32_t> token_ids,
        bool skip_special_tokens = true
    ) const;

    /**
     * Decode multiple token sequences (batch)
     *
     * @param token_sequences Vector of token ID vectors
     * @param skip_special_tokens Skip special tokens in output
     * @return Vector of decoded texts
     */
    std::vector<std::string> decode_batch(
        std::span<const std::vector<uint32_t>> token_sequences,
        bool skip_special_tokens = true
    ) const;

    /**
     * Get vocabulary size
     *
     * @return Number of tokens in vocabulary
     */
    [[nodiscard]] size_t vocab_size() const noexcept {
        return vocab_.size();
    }

    /**
     * Get special tokens
     *
     * @return Special token IDs
     */
    [[nodiscard]] const SpecialTokens& special_tokens() const noexcept {
        return special_tokens_;
    }

    /**
     * Convert token to string
     *
     * @param token_id Token ID
     * @return Token string, or nullopt if invalid
     */
    [[nodiscard]] std::optional<std::string> token_to_string(uint32_t token_id) const;

    /**
     * Convert string to token
     *
     * @param token Token string
     * @return Token ID, or nullopt if not in vocabulary
     */
    [[nodiscard]] std::optional<uint32_t> string_to_token(const std::string& token) const;

private:
    // Vocabulary: token string → ID
    std::unordered_map<std::string, uint32_t> vocab_;

    // Reverse vocabulary: ID → token string
    std::unordered_map<uint32_t, std::string> vocab_reverse_;

    // BPE merge rules (sorted by priority)
    std::vector<BPEMerge> bpe_merges_;

    // Special tokens
    SpecialTokens special_tokens_;

    /**
     * Apply BPE merges to text
     *
     * @param text Input text
     * @return Byte pairs after merging
     */
    std::vector<std::string> apply_bpe(std::string_view text) const;

    /**
     * Byte-level pre-tokenization
     *
     * @param text Input text
     * @return Byte-level tokens
     */
    static std::vector<uint8_t> byte_encode(std::string_view text);

    /**
     * Byte-level decoding
     *
     * @param bytes Byte-level tokens
     * @return Decoded text
     */
    static std::string byte_decode(std::span<const uint8_t> bytes);

    /**
     * Validate UTF-8 encoding
     *
     * @param text Text to validate
     * @return true if valid UTF-8
     */
    static bool is_valid_utf8(std::string_view text) noexcept;

    /**
     * Load tokenizer from JSON file
     *
     * @param path Path to tokenizer JSON
     */
    void load_from_json(const std::string& path);
};

/**
 * Convenience function to load tokenizer
 *
 * @param tokenizer_path Path to gremlin_tokenizer.json
 * @return Loaded tokenizer
 */
inline GremlinTokenizer load_tokenizer(const std::string& tokenizer_path) {
    return GremlinTokenizer(tokenizer_path);
}

} // namespace gremlin
