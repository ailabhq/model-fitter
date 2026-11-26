import sys, gguf
r = gguf.GGUFReader(sys.argv[1])
t = next(x for x in r.tensors if x.name == sys.argv[2])
sys.stdout.buffer.write(t.data.tobytes())
