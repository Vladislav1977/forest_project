import argparse
import sys
import ast

def string_parse(str_arg):
    try:
        str_arg = ast.literal_eval(str_arg)
        return str_arg
    except:
        return str_arg

def parse_unknown(args):
    key = args[::2]
    val = list(map(lambda x: string_parse(x), args[1::2]))
    return dict(zip(key, val))




parser = argparse.ArgumentParser()
parser.add_argument('--foo', help='foo help')
#parser.add_argument('d', help='foo help')

args, unknown = parser.parse_known_args()

print("foo", args.foo)
print("unknown", unknown)
print("unknown", parse_unknown(unknown))
print(sys.argv)
#print("d", args.d)