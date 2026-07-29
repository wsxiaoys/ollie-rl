#!/usr/bin/env bash

set -u

expected=/logs/verifier/expected-answer.txt
printf 'ollie harbor smoke test\n' > "$expected"

if cmp -s "$expected" /workspace/answer.txt; then
  echo 1 > /logs/verifier/reward.txt
  echo "smoke test passed"
else
  echo 0 > /logs/verifier/reward.txt
  echo "expected /workspace/answer.txt to contain exactly: ollie harbor smoke test" >&2
fi
