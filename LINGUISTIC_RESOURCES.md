# GREMLIN Linguistic Resources

This document lists all the linguistic resources and parameters available in the GREMLIN language pack generator.

## 📚 Concept Dictionary

**Total Concepts:** 186

### Categories:

1. **Actions** (50 concepts)
   - Core verbs: send, receive, authenticate, verify, encrypt, etc.
   - Communication: ask, answer, confirm, deny, request, etc.
   - System actions: start, stop, pause, resume, reset, etc.

2. **Identity** (20 concepts)
   - Pronouns: I, you, we, they
   - Roles: client, server, admin, user, system
   - Relationships: friend, stranger, authority

3. **Connectors** (10 concepts)
   - Logical: and, or, not, if, then, else
   - Temporal: before, after, during, while

4. **Grammar Particles** (6 concepts)
   - Question markers, negation, emphasis
   - Tense/aspect indicators

5. **Variables - Names** (50 concepts)
   - Common first names across cultures
   - Used for authentication challenges

6. **Variables - Companies** (50 concepts)
   - Well-known company names
   - Used for authentication challenges

7. **Variables - Topics** (50 concepts)
   - Common subjects and themes
   - Used for authentication challenges

## 🌍 Grammar Systems

**6 Word Orders from World Languages:**

| Grammar | Pattern | Example Languages | Speakers |
|---------|---------|-------------------|----------|
| **SVO** | Subject-Verb-Object | English, Chinese, French, Spanish | ~2.4 billion |
| **SOV** | Subject-Object-Verb | Japanese, Korean, Turkish, Hindi | ~1.5 billion |
| **VSO** | Verb-Subject-Object | Irish, Arabic, Filipino | ~200 million |
| **VOS** | Verb-Object-Subject | Malagasy, Fijian | ~25 million |
| **OVS** | Object-Verb-Subject | Hixkaryana, Urubú-Kaapor | ~1,000 |
| **OSV** | Object-Subject-Verb | Warao, Nadëb | ~30,000 |

**Total Coverage:** Represents ~4.1+ billion speakers across all continents

## 🔤 Unicode Character Blocks

**22 Writing Systems + Symbol Sets:**

### Alphabetic Scripts (7)
1. **Latin Basic** (0x0041-0x007A)
   - English, Spanish, French, German, etc.
2. **Latin Extended-A** (0x0100-0x017F)
   - European languages with diacritics
3. **Latin Extended-B** (0x0180-0x024F)
   - African, Vietnamese, Pinyin
4. **Cyrillic** (0x0400-0x04FF)
   - Russian, Ukrainian, Bulgarian, Serbian
5. **Greek** (0x0370-0x03FF)
   - Greek language, mathematical notation
6. **Arabic** (0x0600-0x06FF)
   - Arabic, Persian, Urdu
7. **Hebrew** (0x0590-0x05FF)
   - Hebrew, Yiddish

### Abugida Scripts (3)
8. **Devanagari** (0x0900-0x097F)
   - Hindi, Sanskrit, Marathi, Nepali
9. **Bengali** (0x0980-0x09FF)
   - Bengali, Assamese
10. **Thai** (0x0E00-0x0E7F)
    - Thai, Lao

### Syllabary Scripts (3)
11. **Ethiopic** (0x1200-0x137F)
    - Amharic, Tigrinya
12. **Cherokee** (0x13A0-0x13FF)
    - Cherokee language
13. **Hiragana** (0x3040-0x309F)
    - Japanese phonetic script
14. **Katakana** (0x30A0-0x30FF)
    - Japanese phonetic script (foreign words)

### Logographic Scripts (2)
15. **CJK Unified** (0x4E00-0x9FFF)
    - Chinese, Japanese, Korean (20,992 characters!)
16. **Hangul Syllables** (0xAC00-0xD7AF)
    - Korean alphabet combinations (11,172 syllables!)

### Symbol Sets (5)
17. **Mathematical Symbols** (0x2200-0x22FF)
    - ∀, ∃, ∈, ∑, ∏, ∫, ≈, ≠, ≤, ≥
18. **Arrows** (0x2190-0x21FF)
    - ←, →, ↑, ↓, ↔, ⇒, ⇔
19. **Miscellaneous Symbols** (0x2600-0x26FF)
    - ☀, ☁, ☂, ☃, ★, ☆, ♠, ♣, ♥, ♦
20. **Geometric Shapes** (0x25A0-0x25FF)
    - ■, □, ▲, △, ●, ○, ◆, ◇
21. **Emoji Basic** (0x1F300-0x1F5FF)
    - Weather, animals, objects
22. **Emoji People** (0x1F600-0x1F64F)
    - Faces, gestures, people

**Total Character Space:** 54,000+ unique Unicode characters available!

## 🎛️ Generation Parameters

### Words per Concept
- **Range:** 100 - 50,000
- **Presets:**
  - Tiny: 500 words (~4 MB, 93K total)
  - Small: 1,000 words (~8 MB, 186K total)
  - Medium: 5,000 words (~39 MB, 930K total)
  - Large: 10,000 words (~78 MB, 1.86M total)
  - Huge: 25,000 words (~195 MB, 4.65M total)
  - Ultra: 50,000 words (~390 MB, 9.3M total)

### Word Length
- **Min Length:** 2-20 characters
- **Max Length:** 2-30 characters
- **Default:** 4-15 characters
- **Recommendation:** Longer = more unique, Shorter = more readable

### Selection Presets
- **Select All:** All 22 Unicode blocks
- **Select None:** Clear all selections
- **Latin Only:** Latin Basic + Extended A + Extended B
- **Diverse Mix:** 10 blocks spanning Latin, Cyrillic, Greek, Arabic, Hebrew, CJK, Hiragana, Katakana, Math symbols

## 🔢 Capacity Calculations

### Authentication Rounds
Each authentication exchange uses ~8 words (both directions).

| Words/Concept | Total Words | Auth Rounds |
|--------------|-------------|-------------|
| 500 | 93,000 | ~11,625 |
| 1,000 | 186,000 | ~23,250 |
| 5,000 | 930,000 | ~116,250 |
| 10,000 | 1,860,000 | ~232,500 |
| 25,000 | 4,650,000 | ~581,250 |
| 50,000 | 9,300,000 | ~1,162,500 |

### File Sizes
Approximate JSON file size: **~0.042 bytes per word**

## 🎨 Example Outputs

### Filename Format
```
language_pack_{words}w_{grammar}_{name}_{timestamp}.json
```

### Examples
```
language_pack_10000w_SVO_20251117_235959.json
language_pack_5000w_SOV_JapaneseStyle_20251118_001234.json
language_pack_500w_VSO_TestPack_20251118_002345.json
```

## 🔐 Cryptographic Properties

### One-Time Pad System
- Each word used **exactly once** then burned
- Words marked as: `unused` → `used` → `use_last` (DDoS pool)
- No word ever appears twice in communication
- Perfect forward secrecy on language rotation

### Entropy Calculations
With 10,000 words per concept:
- **Per-word entropy:** log₂(10,000) ≈ 13.3 bits
- **Per-sentence entropy:** ~80-106 bits (6-8 words)
- **Attack resistance:** 2^80 to 2^106 combinations per message

### Unicode Diversity
Using all 22 blocks (54,000+ chars):
- Character space larger than entire English alphabet
- Multi-lingual obfuscation prevents pattern recognition
- Emoji/symbols prevent linguistic analysis tools

## 📖 Linguistic Theory

GREMLIN implements concepts from:

1. **Information Theory** (Shannon)
   - One-time pads for perfect secrecy
   - Entropy maximization

2. **Generative Grammar** (Chomsky)
   - Universal grammar patterns
   - Parameterized word order

3. **Typology** (Greenberg)
   - Cross-linguistic universals
   - Word order implicational hierarchies

4. **Semiotics** (Saussure)
   - Arbitrary sign-meaning relationships
   - Ephemeral linguistic systems

5. **Cryptolinguistics**
   - Navajo Code Talkers (WW2)
   - Constructed languages for security

## 🎯 Use Cases

### Security Applications
- AI-to-AI authentication
- Zero-knowledge proofs
- Ephemeral messaging
- DDoS protection
- MITM resistance

### Accessibility Applications
- Aphasia patient communication
- Locked-in syndrome (Bruce Willis case)
- Alternative communication systems
- Cognitive flexibility training

---

**Total Linguistic Diversity:** 4+ billion speakers represented, 54,000+ characters, trillions of possible languages!
