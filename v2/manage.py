"""Administration: account provisioning, consistent backups, and retention cleanup."""
import argparse
from datetime import datetime, timedelta, timezone
import getpass
from .auth import create_user
from .config import Settings
from .storage import Store


def main():
    parser = argparse.ArgumentParser(description='DataCorn workspace administration')
    commands = parser.add_subparsers(dest='command', required=True)
    account = commands.add_parser('create-user'); account.add_argument('username')
    backup = commands.add_parser('backup'); backup.add_argument('destination')
    prune = commands.add_parser('prune-demo'); prune.add_argument('--days', type=int, default=30)
    args = parser.parse_args()
    store = Store(Settings.from_env().data_dir)
    if args.command == 'create-user':
        password = getpass.getpass('New password (12+ characters): ')
        if password != getpass.getpass('Repeat password: '):
            parser.error('Passwords do not match.')
        create_user(store, args.username, password)
        print('Private account created.')
    elif args.command == 'backup':
        print('Backup created:', store.backup(args.destination))
    else:
        if args.days < 1:
            parser.error('Retention must be at least one day.')
        cutoff = (datetime.now(timezone.utc) - timedelta(days=args.days)).isoformat()
        with store.connect() as db:
            count = db.execute("DELETE FROM datasets WHERE owner LIKE 'demo:%' AND created_at<? AND id NOT IN (SELECT dataset_id FROM jobs WHERE status IN ('queued','running'))", (cutoff,)).rowcount
        print('Expired demo workspaces removed:', count)


if __name__ == '__main__':
    main()
