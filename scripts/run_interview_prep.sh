#!/usr/bin/env bash
# Run all four interview prep PDF scripts sequentially.
# Output PDFs land in strats/ (project root).
set -e

cd "$(dirname "$0")/.."
source venv/bin/activate

echo "=== [1/4] SOFR & Fed Rate Path ==="
python scripts/sofr_implied_path.py --out sofr_implied_path.pdf

echo ""
echo "=== [2/4] Rate Differentials & FX ==="
python scripts/rate_differentials_fx.py --out rate_differentials_fx.pdf

echo ""
echo "=== [3/4] MOVE & Rate Vol ==="
python scripts/move_rate_vol.py --out move_rate_vol.pdf

echo ""
echo "=== [4/4] BoE MPC Votes ==="
python scripts/boe_mpc_votes.py --out boe_mpc_votes.pdf

echo ""
echo "Done. PDFs written to:"
ls -lh sofr_implied_path.pdf rate_differentials_fx.pdf move_rate_vol.pdf boe_mpc_votes.pdf
