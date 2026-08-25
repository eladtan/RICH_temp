#!/usr/bin/env python3

from pathlib import Path


def main():
    return Path("test_passed.res").exists() and not Path("test_failed.res").exists()


if __name__ == "__main__":
    Path("test_passed.res" if main() else "test_failed.res").touch()
