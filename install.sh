#!/usr/bin/env bash

isaacgym_dir="/root/isaacgym/"

if [[ ! -d "$isaacgym_dir" ]]; then
  echo "Isaac Gym directory not found at '$isaacgym_dir'. Skipping optional install."
  exit 0
fi

python3 -m pip install -e . --no-deps
python3 -m pip install -e "$isaacgym_dir" --no-deps
