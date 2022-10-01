from sys import stderr

SUCCESS = 0
FAIL = 1

def test_runner(*tests):
    failed = 0
    for test in tests:
        failed += 1 if test() == FAIL else 0
    if not failed:
        print("Passed all tests.")
    else:
        print(f"Got {failed} fails at total. Passed {len(tests) - failed}/{len(tests)} at total.", file = stderr)
