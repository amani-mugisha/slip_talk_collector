import redis 

#Connection to redis
r = redis.Redis(host='localhost' , port=6379, decode_responses=True)

#Adding different url
r.lpush("urls", "https://openai.com")

try:
    r.ping()

    print("Redis connected successfully!")

except redis.exceptions.error as e:

    print(f"Connection error: {e}")

#Detting the urls
url = r.rpop('urls')

if (url):
    print(f"Processing... {url}")

else:
    print('No url found')