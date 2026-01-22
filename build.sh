#!/bin/bash
export LIBCLANG_PATH="/opt/homebrew/opt/llvm/lib"
export HF_HOME="./models_cache"
cargo build --release "$@"
