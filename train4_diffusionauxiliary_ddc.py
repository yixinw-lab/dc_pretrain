"""Compatibility wrapper for the Diffusion-data-constrained (DDC) trainer
adopted from https://arxiv.org/abs/2507.15857.

The implementation lives in ``dcp.train4_diffusionauxiliary_ddc_impl`` so
existing launch scripts can keep invoking this filename unchanged.
"""

import runpy


def main():
    runpy.run_module("dcp.train4_diffusionauxiliary_ddc_impl", run_name="__main__")


if __name__ == "__main__":
    main()
