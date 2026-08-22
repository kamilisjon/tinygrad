#!/usr/bin/env bash
set -euo pipefail
echo -e "\033[1;36m======== 1/2  CAPTURE on hardware ========\033[0m"
VIZ=2 PYTHONPATH=. PROFILE=1 SQTT=1 DEV=PCI+AMD python3 test/amd/test_cycle_accurate_emu.py TestSQTTCapture -v
echo -e "\033[1;36m======== 2/2  COMPARE against emulator ========\033[0m"
PYTHONPATH=. PROFILE=1 DEV=MOCKKFD+AMD python3 test/amd/test_cycle_accurate_emu.py TestSQTTEmu -v
