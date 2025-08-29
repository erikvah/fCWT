import time

__t = 0

def tick():
    global __t
    __t = time.time()

def tock(s):
    global __t
    t_new = time.time()
    print(f"Lap \"{s}\": {(t_new - __t) * 1000} ms")
    __t = t_new