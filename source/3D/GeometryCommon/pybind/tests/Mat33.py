import test_utils
import mat33
import itertools

def test_determinant():
    matrix = mat33.Mat33_int(1, 0, 3, 4, 5, 2, 7, 8, -1)
    det = matrix.determinant()
    if det != -30:
        print(f"FAILURE: Matrix's determinant (determinant()) should be -30, while its {det}")
        return test_utils.FAIL
    det = matrix.det()
    if matrix.det() != -30:
        print(f"FAILURE: Matrix's determinant (det()) should be -30, while its {det}")
        return test_utils.FAIL
    return test_utils.SUCCESS

def test_at():
    original_matrix = [[1, 0, 3], [4, 5, 2], [7, 8, -1]]
    matrix = mat33.Mat33_int(*list(itertools.chain.from_iterable(original_matrix)))
    at_matrix = [[matrix.at(i, j) for j in range(3)] for i in range(3)]
    for i in range(3):
        for j in range(3):
            if matrix.at(i, j)  != original_matrix[i][j]:
                print(f"FAILURE: Matrix's ({i}, {j}) element should be {original_matrix[i][j]}, while its {at_matrix[i][j]}")
                return test_utils.FAIL
    return test_utils.SUCCESS

if __name__ == "__main__":
    print("Mat33's version:", mat33.__version__)
    test_utils.test_runner(test_determinant, test_at)
    # help(mat33.Mat33_int)