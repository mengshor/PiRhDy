from functions import generate_next_phrase_melody, generate_accompaniment
import argparse

parser = argparse.ArgumentParser(description='input attributes')
# name_list = ['0','1','2','3','4','5','6','7','8','9','a','b','c','d','e','f']
parser.add_argument('--name', '-n', default="0")
parser.add_argument('--task_name', '-t', default="next")
args = parser.parse_args()

if __name__ == '__main__':

    try:
        names = args.name
        task_name = args.task_name
        names_list = names.strip("\"").split(',')
        task_list = {"next": generate_next_phrase_melody,
                     "acc": generate_accompaniment}
        for name in names_list:
            task_list[task_name](str(name))
    except Exception as e:
        print(e)
