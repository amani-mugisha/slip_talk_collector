import redis

class RedisQueue:
    def __init__(self):
        self.redis = redis.Redis(
            host='localhost',
            port=6379,
            decode_responses=True
        )

    def push(self, queuename, item):
        return self.redis.lpush(queuename, item)
    
    def pop(self, queuename):
        return self.redis.rpop(queuename)
    
    def size(self, queuename):
        return self.redis.llen(queuename)

queue = RedisQueue()

queue.push()