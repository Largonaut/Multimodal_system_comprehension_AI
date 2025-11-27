/**
 * Trap Door - Character-Level Fallback
 *
 * Handles unknown words by encoding character-by-character.
 * Ensures 100% coverage (zero English leakage).
 *
 * Security: Local encoding only, no network, complete control.
 * Performance: Pre-computed lookup table, sub-microsecond encoding.
 *
 * Author: GREMLIN Team
 * License: MIT
 */

#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <array>

namespace gremlin {

/**
 * Trap Door
 *
 * Character-level encoding for words not in dictionary.
 * Guarantees no English leakage (100% coverage).
 */
class TrapDoor {
public:
    /**
     * Constructor - initializes character mappings
     */
    TrapDoor();

    /**
     * Encode single character to GREMLIN
     *
     * @param c Character to encode
     * @return GREMLIN code for character
     */
    [[nodiscard]] std::string encode_char(char c) const;

    /**
     * Encode single Unicode codepoint
     *
     * @param codepoint Unicode codepoint
     * @return GREMLIN code for codepoint
     */
    [[nodiscard]] std::string encode_codepoint(uint32_t codepoint) const;

    /**
     * Encode entire word character-by-character
     *
     * @param word Word to encode
     * @return GREMLIN-encoded word
     */
    [[nodiscard]] std::string encode_word(std::string_view word) const;

    /**
     * Decode GREMLIN character code
     *
     * @param code GREMLIN character code
     * @return Decoded character, or '\0' if invalid
     */
    [[nodiscard]] char decode_char(std::string_view code) const;

    /**
     * Decode entire GREMLIN word
     *
     * @param gremlin_word GREMLIN-encoded word
     * @return Decoded English word
     */
    [[nodiscard]] std::string decode_word(std::string_view gremlin_word) const;

    /**
     * Check if code is a character-level encoding
     *
     * @param code GREMLIN code
     * @return true if character-level encoding
     */
    [[nodiscard]] bool is_trap_door_code(std::string_view code) const noexcept;

private:
    // ASCII → GREMLIN mapping (0-127)
    std::array<std::string, 128> ascii_to_gremlin_;

    // Extended character → GREMLIN mapping (Unicode)
    std::unordered_map<uint32_t, std::string> extended_to_gremlin_;

    // Reverse mappings
    std::unordered_map<std::string, char> gremlin_to_ascii_;
    std::unordered_map<std::string, uint32_t> gremlin_to_extended_;

    /**
     * Initialize ASCII mappings
     */
    void initialize_ascii_mappings();

    /**
     * Initialize extended (Unicode) mappings
     */
    void initialize_extended_mappings();

    /**
     * Get next Unicode codepoint from UTF-8 string
     *
     * @param utf8 UTF-8 string
     * @param offset Current offset (updated)
     * @return Unicode codepoint
     */
    static uint32_t get_next_codepoint(std::string_view utf8, size_t& offset);

    /**
     * Convert Unicode codepoint to UTF-8
     *
     * @param codepoint Unicode codepoint
     * @return UTF-8 encoded string
     */
    static std::string codepoint_to_utf8(uint32_t codepoint);
};

} // namespace gremlin
