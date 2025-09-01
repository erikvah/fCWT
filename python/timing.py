import time

__t = 0

def tick():
    global __t
    __t = time.time()

def tock(s, supress=False):
    global __t
    t_new = time.time()
    dur = (t_new - __t) * 1000
    if not supress:
        print(f"Lap \"{s}\": {dur} ms")
    __t = t_new

    return dur