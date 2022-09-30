from sys import stderr
import mat33
import itertools

FAIL = 1
SUCCESS = 0

def test_determinant():
    matrix = mat33.Mat33_int(1, 0, 3, 4, 5, 2, 7, 8, -1)
    det = matrix.determinant()
    if det != -30:
        print(f"FAILURE: Matrix's determinant (determinant()) should be -30, while its {det}")
        return FAIL
    det = matrix.det()
    if matrix.det() != -30:
        print(f"FAILURE: Matrix's determinant (det()) should be -30, while its {det}")
        return FAIL
    return SUCCESS

def test_at():
    original_matrix = [[1, 0, 3], [4, 5, 2], [7, 8, -1]]
    matrix = mat33.Mat33_int(*list(itertools.chain.from_iterable(original_matrix)))
    at_matrix = [[matrix.at(i, j) for j in range(3)] for i in range(3)]
    for i in range(3):
        for j in range(3):
            if matrix.at(i, j)  != original_matrix[i][j]:
                print(f"FAILURE: Matrix's ({i}, {j}) element should be {original_matrix[i][j]}, while its {at_matrix[i][j]}")
                return FAIL
    return SUCCESS

def test_runner(*tests):
    failed = 0
    for test in tests:
        failed += 1 if test() == FAIL else 0
    if not failed:
        print("Passed all tests.")
    else:
        print(f"Got {failed} fails at total. Passed {len(tests) - failed}/{len(tests)} at total.", file=stderr)

if __name__ == "__main__":
    print("version:", mat33.__version__)
    test_runner(test_determinant, test_at)