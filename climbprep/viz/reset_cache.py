import time
import diskcache
import argparse

from climbprep.util import *
from climbprep.viz.app import CACHE_PATH, CACHE_SIZE

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='Reset the disk cache used by `climbprep.viz`.')
    argparser.add_argument('keys', nargs='*', help='Optional keys to reset specific cache entries. '
                                                   'IMPORTANT: Uses partial string matching. If no keys are '
                                                   'provided, the entire cache will be cleared.')
    args = argparser.parse_args()
    keys = args.keys

    cache = diskcache.Cache(
        CACHE_PATH,
        size_limit=CACHE_SIZE
    )
    t0 = time.time()
    if keys:
        unique_keys = set()
        for entry in cache.iterkeys():
            unique_keys.add(entry[0])
        for key in unique_keys:
            for del_key in keys:
                if del_key in key:
                    stderr(f'Removing cache entry: {key}\n')
                    cache.delete(key)
    else:
        stderr(f'Clearing entire cache at {CACHE_PATH}...\n')
        cache.clear()
    t1 = time.time()
    stderr(f'  Finished in {t1 - t0: 0.2f} seconds.\n')