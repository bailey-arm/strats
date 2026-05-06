#!/bin/bash
cd "$(dirname "$0")"
source venv/bin/activate
python3 scripts/crypto_live.py
