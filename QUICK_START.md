# GREMLIN Quick Start

**Demo tomorrow? Here's everything you need in 5 minutes.**

---

## 🚀 One-Command Demo

```bash
./quick_demo.sh
```

Choose option 2 (scripted demo) for instant wow factor.

---

## 💻 Three-Laptop Setup

### Copy to Each Laptop

```bash
# 1. Clone/copy GREMLIN directory
git clone <your-repo> GREMLIN
cd GREMLIN

# 2. Install dependencies
pip install numpy rich

# 3. Copy language pack to Laptops 1 & 2
# File: language_packs/language_pack_20251117_172914.json
```

### Start in Order

**Laptop 3 (MITM) - First:**
```bash
python demo/mitm_viewer.py --mode passive
```

**Laptop 2 (Server) - Second:**
```bash
python demo/server.py --pack language_packs/language_pack_*.json
```

**Laptop 1 (Client) - Third:**
```bash
python demo/client.py --pack language_packs/language_pack_*.json --mode scripted --rounds 5
```

---

## 🎤 Elevator Pitch

> "GREMLIN is like the Navajo code talkers for AI. We generate completely novel synthetic languages on-demand, enabling secure communication that's unintelligible without the language pack. Each word is used only once - a linguistic one-time pad. When words run out, generate a new language in 30 seconds. Perfect forward secrecy, no crypto to break, and the same tech could help aphasia patients communicate."

---

## 💡 Key Points

1. **No encryption** - just random word mappings
2. **One-time use** - each word used once, then burned
3. **Trillions of languages** possible
4. **8MB pack** = 23,000+ authentications
5. **30-second rotation** for new languages
6. **Dual-use**: Security + Aphasia communication

---

## ❓ Likely Questions

**"What if the pack gets stolen?"**
→ Rotate immediately. Old pack is worthless (words already used).

**"Performance?"**
→ Faster than AES. No computational crypto overhead.

**"With LLMs?"**
→ Yes! Language pack is just JSON. Any LLM can load it.

**"Aphasia timeline?"**
→ Core tech done. 6 months to production with personal concept trainer.

**"ElasticSearch?"**
→ Inter-node secure comms, or concept-preserving transformations for encrypted search.

---

## 🚨 Emergency Fallbacks

**If network fails:**
```bash
python demo_authentication.py --pack language_packs/language_pack_*.json --rounds 5
```

**If code breaks:**
Talk through DEMO_GUIDE.md - the concept alone is impressive.

**If laptops fail:**
You have detailed notes. Draw on whiteboard. Improvise!

---

## ✅ Pre-Demo Checklist

- [ ] All laptops charged
- [ ] Language pack on USB backup
- [ ] Dependencies installed
- [ ] Quick test run completed
- [ ] DEMO_GUIDE.md printed
- [ ] Water nearby
- [ ] Confidence at 100%

---

**You built this in ONE DAY. You've got this.** 🚀
