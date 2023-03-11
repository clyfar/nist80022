#!/bin/sh
# Check to see if Homebrew, Go, and Pre-commit are installed, and install it if it is not
HAS_BREW=$(command -v brew >/dev/null)
HAS_GO=$(command -v go >/dev/null)
HAS_PRECOMMIT=$(command -v pre-commit >/dev/null)

if $HAS_BREW; then
  echo >&2 "Homebrew is already installed"
else
  echo >&2 "Installing Homebrew Now"; \
  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)";
fi

if $HAS_GO && $HAS_BREW; then
  echo >&2 "Go is already installed"
else
  echo >&2 "Installing Go Now"; \
  brew install go;
fi

if $HAS_PRECOMMIT && $HAS_BREW; then
  echo >&2 "pre-commit is already installed"
else
  echo >&2 "Installing pre-commit Now"; \
  brew install pre-commit;
fi

pre-commit install
