#!/bin/sh

HAS_GO=$(command -v go >/dev/null)

if $HAS_GO; then
  go test ./...
else
  echo >&2 "Go must be installed to test"
fi
