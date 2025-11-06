# KickVision v20.0 — 100-Model Football Prediction Bot

> **100 parallel Monte-Carlo models** → **1 ultra-accurate prediction**

Telegram bot that predicts match outcomes using:
- xG modeling
- Form analysis
- H2H history
- Typo-tolerant team search
- Interactive disambiguation

---

## Features

| Feature | Description |
|-------|-----------|
| `Chelsea vs ManU` | Instant prediction |
| `Chelsae`, `Barca`, `Livarpool` | Typo-proof |
| `1 2` reply | Pick from multiple matches |
| 100 models | Stable, robust probabilities |
| Rate-limited | 3 req / 5 sec |

---

## Setup

```bash
# Clone
git clone https://github.com/Firestorm0609/kickvision-bot.git
cd kickvision-bot

# Install
pip install pyTelegramBotAPI requests numpy

# Add your clubs.zip (required)
# Download from: https://github.com/openfootball/clubs
# Save as: clubs.zip

# Run
python kickvision.py
