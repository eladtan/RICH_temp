import test_utils
import vector3D

def simple_test():
    vec = vector3D.Vector3D(1, 2, 3)
    vec[2] = 5
    print(vec[0], vec[1], vec[2])
    return test_utils.SUCCESS

if __name__ == "__main__":
    print("Vector3D's version:", vector3D.__version__)
    test_utils.test_runner(simple_test)
