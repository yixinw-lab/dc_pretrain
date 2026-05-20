"""Compatibility wrapper for the diffusion-auxiliary trainer.

The implementation lives in ``dcp.train4_diffusionauxiliary_impl`` so existing
launch scripts can keep invoking this filename unchanged.
"""

import runpy


def main():
    runpy.run_module("dcp.train4_diffusionauxiliary_impl", run_name="__main__")


if __name__ == "__main__":
    main()
