import os
import time
import shutil
from redis import Redis
from datetime import datetime

# Initialize Cache


def poll_remove_user_data(redis_client: Redis):
    while True:
        try:
            user_list = redis_client.lrange("users", 0, -1)
            if user_list:
                # print("user_list:", user_list)
                print("Hello World! here are the current redis users' contents.")
                for user in user_list:
                    user_data = redis_client.keys(f"{user}*")
                    if user_data:
                        print("for user number:", user)
                        print("user_data:", user_data)
                        # before removing user data check if theyve been gone long enough
                        # retrieve the users stored time stamp and convert from string to datetime object
                        user_time = datetime.strptime(
                            redis_client.get(f"{user}:latest_timestamp"),
                            "%Y-%m-%d %H:%M:%S.%f",
                        )
                        # get current time
                        curr_time = datetime.now()
                        # compare the times
                        time_difference = (curr_time - user_time).total_seconds()
                        print("user_time", user_time)
                        print("curr_time", curr_time)
                        print("difference", time_difference, "sec")
                        if time_difference >= 60:
                            print(f"expired, DELETING {user} DATA!")
                            # Remove user data from redis
                            redis_client.delete(*user_data)
                            # remove userid from users list in redis
                            redis_client.lrem("users", 0, user)
                            # Remove user data from filesystem
                            try:
                                data_directory = os.path.join(
                                    os.getcwd(), f"assets/{user}"
                                )
                                if os.path.exists(data_directory):
                                    shutil.rmtree(data_directory)
                            except Exception as e:
                                # should be logging rather than printing these errors
                                print("[Exception]:", e)
            else:
                print("currently no users!")
        except Exception as e:
            # should be logging rather than printing these errors
            print("[Exception]:", e)
        time.sleep(40)
