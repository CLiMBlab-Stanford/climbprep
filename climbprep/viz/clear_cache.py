import os
import shutil

cache_path = os.path.join(os.getcwd(), '.cache', 'viz')
if os.path.exists(cache_path):
    shutil.rmtree(cache_path)
    print(f"Cache cleared at {cache_path}")