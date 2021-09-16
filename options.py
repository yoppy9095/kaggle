import argparse

def opt_args():
    parser = argparse.ArgumentParser(description='titannic')
    parser.add_argument(
        '--train_path',
        default='D:/kaggle/titanic/train.csv'
    )
    parser.add_argument(
        '--test_path',
        default='D:/kaggle/titanic/test.csv'
    )





    args = parser.parse_args()
    
    return args 