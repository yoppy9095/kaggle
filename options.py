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
    parser.add_argument(
        '--model_name',
        default='random_forest'
    )
    parser.add_argument(
        '--lr',
        default=1e-4
    )
    parser.add_argument(
        '--batch_size',
        default=32
    )
    parser.add_argument(
        '--epochs',
        default=800
    )



    args = parser.parse_args()
    
    return args 