#!/usr/bin/env bash

set -u

if bun /tests/test_state.ts; then
  echo 1 > /logs/verifier/reward.txt
else
  echo 0 > /logs/verifier/reward.txt
fi
