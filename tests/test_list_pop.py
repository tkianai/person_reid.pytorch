
import time


data = list(range(1000000))

start = time.time()
while len(data) != 0:
    data.pop(0)

end = time.time()
print("pop first costs: {}".format(end - start))

start = time.time()
while len(data) != 0:
    data.pop()

end = time.time()
print("pop last costs: {}".format(end - start))
