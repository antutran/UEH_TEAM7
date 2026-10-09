#!/usr/bin/env python3
import os, pty, select, sys

def run(cmd):
    pid, fd = pty.fork()
    if pid == 0:
        os.execvp('ssh', ['ssh', '-tt', '-o', 'StrictHostKeyChecking=no', 'jetson@172.20.10.2', cmd])
    else:
        output = b''
        while True:
            r, _, _ = select.select([fd], [], [], 10)
            if not r: break
            try:
                data = os.read(fd, 2048)
                if not data: break
                output += data
                sys.stdout.buffer.write(data)
                sys.stdout.buffer.flush()
                if b'password:' in output.lower():
                    os.write(fd, b'jetson\n')
                    output = b''
            except OSError:
                break

if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'echo "hello"'
    run(cmd)
