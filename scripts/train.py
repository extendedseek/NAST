#!/usr/bin/env python3
import sys

from nast.cli import main


if __name__ == "__main__":
    main(["train", *sys.argv[1:]])
