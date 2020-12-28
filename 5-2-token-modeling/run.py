from settings import run_script
import sys
import argparse

parser = argparse.ArgumentParser(description='input attributes')
parser.add_argument('--gpu_names', '-g', default=None, help='the ids of gpus, '
                                                            '\"None (default)\" for cpu running,'
                                                            '\"0,1\" for using multiple gpus')
parser.add_argument('--model_name', '-m', default='overall', help='model names involve '
                                                                  'overall, '
                                                                  'overall_con, '
                                                                  'chroma, '
                                                                  'chroma_octave, '
                                                                  'chroma_position, '
                                                                  'chroma_velocity, '
                                                                  'chroma_state, '
                                                                  'melody, harmony, '
                                                                  'chroma_melody, chroma_harmony, '
                                                                  'chroma_octave_melody, chroma_octave_harmony'
                                                                  'chroma_velocity_melody, chroma_velocity_harmony'
                                                                  'chroma_position_melody,'
                                                                  'chroma_state_melody, chroma_state_harmony')
parser.add_argument('--ratio', '-r', default=100, help='the ratio(0-100) of training dataset, e.g., 80% of data, -r=80')
args = parser.parse_args()

if __name__ == '__main__':
    try:
        run_script(gpu_names=args.gpu_names, model_name=args.model_name, ratio=args.ratio)
    except Exception as e:
        print(e)