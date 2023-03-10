#!/bin/sh
# Check to see if Homebrew is installed, and install it if it is not
HAS_BREW=$(command -v brew >/dev/null)

if $HAS_BREW; then
  echo >&2 "Homebrew is already installed"
else
  echo >&2 "Installing Homebrew Now"; \
  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)";
  brew install golang
  brew install pre-commit
fi

pre-commit install
