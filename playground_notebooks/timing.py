from datetime import datetime
import time
from redis import Redis

redis_client = Redis(host="redis", port=6379, db=0, decode_responses=True)

redis_client.set(f"uuid1:latest_timestamp", str(datetime.now()))
time.sleep(5)
reverted_time = datetime.strptime(
    redis_client.get(f"uuid1:latest_timestamp"), "%Y-%m-%d %H:%M:%S.%f"
)

print("reverted:", type(reverted_time))
print(reverted_time)
time.sleep(3.5)
new_time = datetime.now()
if new_time > reverted_time:
    print(new_time)
    print("expired")
# finish finding the format and convert there to string and back
