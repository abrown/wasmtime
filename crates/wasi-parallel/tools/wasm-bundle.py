import argparse
import os.path
import shutil
import sys
import tempfile

# Print an error and terminate program
def terminate(msg):
    print(f"Error: {msg}", file=sys.stderr)
    sys.exit(1)

parser = argparse.ArgumentParser(description="Bundle a WebAssembly module inside of a WebAssembly object file")
parser.add_argument("symbol", help="Symbol name pointing to embedded byte array")
parser.add_argument("input", help="Path to Wasm module to be embedded")
parser.add_argument("-o", "--output", help="Path to output object file")
parser.add_argument("--size-symbol", help="Name for size symbol")
parser.add_argument("-v", "--verbose", help="Turn on verbose mode", action="store_true")
args = parser.parse_args()

# Implied values
out_file = args.input + ".o"
size_sym = args.symbol + "_size"

# Override implied values
if args.output:
    out_file = args.output
if args.size_symbol:
    size_sym = args.size_symbol

# Find Clang on PATH
cc = shutil.which("clang")

if (cc is None):
    terminate("Could not find clang")
    # TODO accept emscripten as well?

if args.verbose:
    print(f"Using clang: {cc}")

# Read module file
if not os.path.exists(args.input):
    terminate(f"Could not open `{args.input}`")
    
with open(args.input, mode='rb') as file:
    input_binary = file.read()
size = len(input_binary)

if args.verbose:
    print(f"Read {size} bytes from {args.input}")

# Print otput to a temp file
fd, tmp = tempfile.mkstemp(suffix = ".c", text=True)
with open(tmp, mode="w") as temp_src:
    temp_src.write(f"unsigned long {size_sym} = {size};\n")
    temp_src.write(f"unsigned char {args.symbol}[{size}] = {{")
    first = True
    for b in input_binary:
        if first:
            first = False
        else:
            temp_src.write(", ")
        temp_src.write(f"{hex(b)}")
    temp_src.write(f"}};")
os.close(fd)

# Compile to an object file
cc_cmd = f"{cc} --target=wasm32 -c {tmp} -o {out_file}"
if args.verbose:
    print(cc_cmd)
ret = os.system(cc_cmd)

# Cleanup the temp file
os.remove(tmp)

# Pass exit status out
sys.exit(ret)

