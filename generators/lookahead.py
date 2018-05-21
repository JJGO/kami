import multiprocessing
import weakref


class LookAhead:

    def __init__(self, iterable, prefetch=1):
        self.iterator = iter(iterable)
        self.q = multiprocessing.Queue(maxsize=prefetch)
        self.p = multiprocessing.Process(target=self._fetcher)
        self.p.start()
        self.finalizer = weakref.finalize(self, self.p.terminate)

    def _fetcher(self):
        while True:
            try:
                x = next(self.iterator)
                self.q.put(x)
            except StopIteration:
                self.q.put(StopIteration)
                return

    def __iter__(self):
        return self

    def __next__(self):
        x = self.q.get()
        if x is StopIteration:
            raise StopIteration
        else:
            return x

    def close(self):
        if self.finalizer is not None:
            self.finalizer()