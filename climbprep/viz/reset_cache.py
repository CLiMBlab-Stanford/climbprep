import time
import diskcache
from climbprep.viz.app import CACHE_PATH, CACHE_SIZE

if __name__ == '__main__':
    cache = diskcache.Cache(
        CACHE_PATH,
        size_limit=CACHE_SIZE
    )
    print(f'Resetting cache at {CACHE_PATH}...')
    t0 = time.time()
    cache.clear()
    t1 = time.time()
    print(f'Cache cleared in {t1 - t0: 0.2f} seconds.')