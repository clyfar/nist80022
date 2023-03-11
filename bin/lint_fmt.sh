#!/bin/sh

HAS_GO=$(command -v go >/dev/null)
HAS_GOLINT=$(command -v golint >/dev/null)

if $HAS_GO && $HAS_GOLINT; then
  golint ./...
  go vet ./...
  go fmt ./...
fi
