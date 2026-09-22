
import sys
data = sys.stdin.buffer.read()
with open(sys.argv[1], 'wb') as f:
    f.write(data)
print(f'Wrote {len(data)} bytes to {sys.argv[1]}')
