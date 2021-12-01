from absl_mock import Mock_Flag, local_test

if __name__ == '__main__':
    local_test()
    flag = Mock_Flag()
    FLAGS = flag.FLAGS
    print(f"my old cars haha : {FLAGS.car} done!!\n")

