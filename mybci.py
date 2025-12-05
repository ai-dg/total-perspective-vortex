import argparse
from logreg import LogReg

def main():
    parser = argparse.ArgumentParser(
        description='Total Perspective Vortex - Brain Computer Interface',
    )
    
    parser.add_argument('subject_id', type=int, nargs='?', 
                       help='Subject ID (1-109)')
    parser.add_argument('run', type=int, nargs='?',
                       help='Run number (1-14)')
    parser.add_argument('mode', type=str, nargs='?', choices=['train', 'predict'],
                       help='Mode: train, predict')

    args = parser.parse_args()



if __name__ == "__main__":
    main()