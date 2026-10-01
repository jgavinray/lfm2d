#!/usr/bin/env bash
# Mandatory CPU + physical GPU gate for changes to device/backend execution.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
backend="${1:?usage: bash demo/test_devices.sh rocm|cuda|metal|sycl}"
case "$backend" in
  'rocm'|'cuda'|'metal'|'sycl') ;;
  *) echo 'Expected rocm, cuda, metal, or sycl (not auto).' >&2; exit 2 ;;
esac
export RAYON_NUM_THREADS='8'
cargo build --release -p 'lfm2d' --features "$backend"
LFM2D_TEST_GPU="$backend" cargo test --release -p 'lfm2d' --features "$backend" \
  --test 'device_real' -- --ignored --nocapture
LFM2D_TEST_DEVICE='cpu' python3 'demo/e2e.py' -v
LFM2D_TEST_DEVICE="$backend" python3 'demo/e2e.py' -v
# Run the same real suite through auto success and auto initialization failure.
LFM2D_TEST_DEVICE='auto' LFM2D_EXPECT_DEVICE="$backend" python3 'demo/e2e.py' -v
LFM2D_TEST_DEVICE='auto' LFM2D_EXPECT_DEVICE='cpu' LFM2D_TEST_DEVICE_INDEX='2147483647' \
  python3 'demo/e2e.py' -v
