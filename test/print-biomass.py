from argparse import ArgumentParser

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("file", type=str)

    args = parser.parse_args()

    l = []
    with open(args.file, 'r') as fp:
        for line in fp:
            l.append(int(line))
    
    print(l)
    