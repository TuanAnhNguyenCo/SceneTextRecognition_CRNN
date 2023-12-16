import argparse
def parse_opts_offline():
    # Offline means not real time 
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--root_path',
        default='SceneTrialTrain',
        type=str,
        help='Root directory path of data')
    parser.add_argument(
        '--data_file_path',
        default='words.xml',
        type=str,
        help='Root directory path of data')
    parser.add_argument(
        '--yolo_data_dir',
        default='yolo_data',
        type=str,
        help='Root directory path of data')
    parser.add_argument(
        '--crnn_data_dir',
        default='crnn_data',
        type=str,
        help='Root directory path of data')
    parser.add_argument(
        '--yolo_data_yaml',
        default='/mnt/disk1/anhnct/School/MachineLearning/Project/SceneTextRecognition',
        type=str,
        help='Root directory path of data')
    parser.add_argument(
        '--train_batch_size',
        default='32',
        type=int,
        help='Root directory path of data')
    parser.add_argument(
        '--val_batch_size',
        default='32',
        type=int,
        help='Root directory path of data')
    parser.add_argument(
        '--test_batch_size',
        default='32',
        type=int,
        help='Root directory path of data')
    parser.add_argument(
        '--hidden_size',
        default='256',
        type=int,
        help='Root directory path of data')
    parser.add_argument(
        '--n_layers',
        default='3',
        type=int,
        help='Root directory path of data')
    parser.add_argument(
        '--dropout_prob',
        default='0.2',
        type=float,
        help='Root directory path of data')
    parser.add_argument(
        '--unfreeze_layers',
        default='32',
        type=int,
        help='Root directory path of data')
    parser.add_argument(
        '--device',
        default='cuda:1',
        type=str,
        help='Root directory path of data')
    parser.add_argument(
        '--threadhold',
        default='0.85',
        type=float,
        help='Root directory path of data')
    args = parser.parse_args()
    return args

