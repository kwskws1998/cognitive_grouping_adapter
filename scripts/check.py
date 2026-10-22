"""Run the complete dependency-light unittest suite from any working directory."""

import sys
import unittest
import argparse

from _common import PROJECT_ROOT


def main():
    """Return a nonzero exit code if any numerical or protocol check fails."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report")
    args = parser.parse_args()
    suite = unittest.defaultTestLoader.discover(str(PROJECT_ROOT / "tests"), pattern="test_*.py")
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if args.report:
        from cog_grouping.provenance import atomic_json
        atomic_json(args.report, {"tests_run": result.testsRun, "failures": len(result.failures),
                                 "errors": len(result.errors), "skipped": len(result.skipped),
                                 "passed": result.testsRun - len(result.failures) - len(result.errors) - len(result.skipped),
                                 "successful": result.wasSuccessful()})
    sys.exit(not result.wasSuccessful())


if __name__ == "__main__":
    main()
