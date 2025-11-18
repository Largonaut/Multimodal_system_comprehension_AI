# GREMLIN Shortcuts

Quick launcher shortcuts for GREMLIN applications.

## Installation

Copy these three files to `F:\dev\Admin_Console_Shortcuts\`:

- **Admin_Console.bat** - Launch the admin console (communication + language viewer)
- **Language_Generator.bat** - Launch the language pack generator
- **Language_Viewer.bat** - Launch the language pack viewer

## Usage

**Option 1: Double-Click**

Simply double-click any of the `.bat` files to launch that application.

**Option 2: Create Desktop Shortcuts**

1. Right-click on any `.bat` file
2. Select "Send to" → "Desktop (create shortcut)"
3. Optionally rename the shortcut (remove ".bat" from the name)

**Option 3: Pin to Taskbar (Windows 11)**

1. Right-click the `.bat` file
2. Select "Pin to taskbar"

## Applications

### 🎮 Admin Console
**File:** `Admin_Console.bat`

- Communication interface
- Send messages as client or server
- View MITM interception
- Language pack browser built-in
- Load/switch language packs

### 🌍 Language Generator
**File:** `Language_Generator.bat`

- Generate new language packs
- 5 tiers: Base (186) → Complete WordNet (117K)
- Full parameter control (words/synset, grammar, Unicode blocks)
- Real-time file size estimates
- Dev mode for massive packs

### 📖 Language Viewer
**File:** `Language_Viewer.bat`

- Browse existing language packs
- Search concepts by English words
- View all synthetic word mappings
- Filter by unused/used/use-last
- Perfect for inspection and verification

## Troubleshooting

**Error: "Cannot find GREMLIN installation"**

The shortcuts are configured for `F:\dev\GREMLIN_Claude_Code_Web_track`.

If your GREMLIN installation is elsewhere, edit the `.bat` file and change this line:
```batch
cd /d F:\dev\GREMLIN_Claude_Code_Web_track
```

**Error: "No module named 'nltk'"**

Install nltk:
```powershell
pip install nltk
python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"
```

**Error: "No module named 'tkinter'"**

Tkinter should come with Python. If missing, reinstall Python with "tcl/tk" option enabled.

---

**All shortcuts navigate to the GREMLIN directory automatically.**
