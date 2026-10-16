#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

"$SCRIPT_DIR/run_pcb_target_fewshot_seed42.sh"
"$SCRIPT_DIR/run_piledbags_target_fewshot_seed42.sh"
