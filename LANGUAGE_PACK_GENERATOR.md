# Language Pack Generator

This standalone module generates GREMLIN synthetic language packs.

## 🎨 GUI Version (Recommended!)

**The easiest way to generate language packs:**

1. Double-click **`run_generator.bat`**
2. Or run: `python language_pack_generator_gui.py`

### GUI Features:
- 🎚️ **Words per Concept Slider** (100-50,000) with presets
- 📏 **Word Length Control** (min/max character range)
- 🌍 **Grammar Selection** (6 world language patterns: SVO, SOV, VSO, VOS, OVS, OSV)
- 🔤 **Unicode Block Selection** (22 writing systems + symbols)
  - Latin, Cyrillic, Greek, Arabic, Hebrew
  - Devanagari, Bengali, Thai, Ethiopic, Cherokee
  - Hiragana, Katakana, CJK, Hangul
  - Mathematical symbols, Arrows, Geometric shapes, Emoji
- 📁 **Output Directory Picker**
- 🏷️ **Optional Language Name** (included in filename)
- 📊 **Real-time Progress** with status log
- 📈 **File Size Estimation**

## Quick Start (Command Line)

### Option 1: Double-Click (Easiest)
1. Double-click `generate_massive_pack.bat`
2. Wait 2-5 minutes
3. Done! A ~78 MB pack with 1.86M words will be in `language_packs/`

### Option 2: PowerShell Script
```powershell
.\generate_massive_pack.ps1
```

### Option 3: Manual Command
```powershell
python generate_language_pack.py --words 10000 --output language_packs/
```

## What Gets Generated

A language pack contains:
- **1,860,000 unique synthetic words** (10,000 per concept)
- **186 semantic concepts** (actions, identity, variables, etc.)
- **~78 MB JSON file** with usage tracking
- **~232,500 authentication rounds** capacity

## File Output

Output files are saved to `language_packs/` with timestamp:
```
language_packs/language_pack_20251117_233719.json
```

## Usage with Admin Console

After generation, run:
```powershell
python demo/admin_console_tk.py --pack language_packs/language_pack_20251117_233719.json
```

Or use wildcard (PowerShell):
```powershell
$pack = (Get-ChildItem language_packs\*.json | Sort-Object LastWriteTime -Descending | Select-Object -First 1).FullName
python demo/admin_console_tk.py --pack $pack
```

## Custom Pack Sizes

**Lightweight (for testing):**
```bash
python generate_language_pack.py --words 500
```
Output: ~4 MB, 93,000 words

**Standard:**
```bash
python generate_language_pack.py --words 5000
```
Output: ~39 MB, 930,000 words

**Massive (for chat/demo):**
```bash
python generate_language_pack.py --words 10000
```
Output: ~78 MB, 1,860,000 words

**Ultra (if you're crazy):**
```bash
python generate_language_pack.py --words 50000
```
Output: ~390 MB, 9,300,000 words

## Grammar Options

Different word orders from world languages:
```bash
python generate_language_pack.py --grammar SVO  # English-style (default)
python generate_language_pack.py --grammar SOV  # Japanese-style
python generate_language_pack.py --grammar VSO  # Irish-style
python generate_language_pack.py --grammar VOS  # Malagasy-style
python generate_language_pack.py --grammar OVS  # Hixkaryana-style
python generate_language_pack.py --grammar OSV  # Warao-style
```

## How It Works

1. **Loads Concept Dictionary**: 186 semantic concepts from `data/base_concepts.json`
2. **Generates Random Unicode Words**: Using 10+ Unicode blocks (Latin, Cyrillic, Greek, Arabic, Hebrew, CJK, etc.)
3. **Creates Mappings**: Each concept gets N random words assigned
4. **Saves with Usage Tracking**: JSON file with `unused`, `used`, `use_last` pools
5. **One-Time Pad System**: Each word used exactly once, then burned

## Standalone Module

This generator is **completely standalone**:
- ✅ No UI dependencies
- ✅ No network required
- ✅ Pure Python + NumPy
- ✅ Outputs plain JSON files
- ✅ Can be run on any machine

## For Tomorrow's Demo

Generate the massive pack now:
```powershell
.\generate_massive_pack.bat
```

This gives you:
- 232,500+ authentication rounds
- Full chat capability
- Real-time language rotation
- Zero chance of running out of words

---

**Generated packs are .gitignore'd** (too large for git). Generate locally on each machine.
