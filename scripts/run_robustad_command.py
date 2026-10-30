#!/usr/bin/env python3
"""Invoke one RobusTAD Click command without importing unrelated subcommands."""

from __future__ import annotations

import sys
import os
import importlib

# Resolve the pinned local panel configuration without changing HOME.
if os.environ.get("ROBUSTAD_CONFIG_DIR"):
    config_module = importlib.import_module("robustad.config")
    config_module.user_config_dir = lambda *args, **kwargs: os.environ["ROBUSTAD_CONFIG_DIR"]

from robustad.assembly import assembly
from robustad.clean import clean
from robustad.lmcc import lmcc
from robustad.boundary import boundary


COMMANDS = {"lmcc": lmcc, "boundary": boundary, "assembly": assembly, "clean": clean}


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        choices = ", ".join(sorted(COMMANDS))
        raise SystemExit(f"usage: {sys.argv[0]} {{{choices}}} [OPTIONS]")
    name = sys.argv[1]
    COMMANDS[name].main(args=sys.argv[2:], prog_name=f"robustad {name}")


if __name__ == "__main__":
    main()
