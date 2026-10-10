"""Run one experiment and clean up only its own process group."""
import os
import signal
import subprocess
import sys


if __name__ == '__main__':
    process = subprocess.Popen(sys.argv[1:], start_new_session=True)
    try:
        code = process.wait()
    finally:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    sys.exit(code)
