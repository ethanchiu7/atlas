import sys

# default 5
SPLIT_NUM = int(sys.argv[1])

# test / train
TYPE = sys.argv[2]

if __name__ == '__main__':
    line_count = 0
    for line in sys.stdin:
        line = line.strip().strip('\n')
        col_list = line.split(' ')
        if col_list[0] not in ['0', '1']:
            continue
        line_count += 1
        if line_count % SPLIT_NUM == 1 and TYPE == 'test':
            print(line)
        elif TYPE == 'train' and line_count % SPLIT_NUM != 1:
            print(line)
