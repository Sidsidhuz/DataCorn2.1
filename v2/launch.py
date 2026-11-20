"""One explicit launcher; unauthenticated local mode always binds to loopback."""
import argparse
import os
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['local', 'demo', 'private'], default=os.getenv('AUTOINSIGHT_MODE', 'demo'))
    parser.add_argument('--port', type=int, default=int(os.getenv('PORT', '8501')))
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535:
        parser.error('Choose a port between 1024 and 65535.')
    os.environ['AUTOINSIGHT_MODE'] = args.mode
    import streamlit.web.cli as cli
    sys.argv = ['streamlit', 'run', str(Path(__file__).with_name('app.py')),
                '--global.developmentMode=false', '--browser.gatherUsageStats=false',
                '--server.maxUploadSize=10', '--server.enableXsrfProtection=true',
                '--server.enableCORS=true', '--server.address=' + ('127.0.0.1' if args.mode == 'local' else '0.0.0.0'),
                '--server.port=' + str(args.port)]
    cli.main()


if __name__ == '__main__':
    main()
