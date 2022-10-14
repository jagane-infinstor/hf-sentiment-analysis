import tweepy as tw
import json
import boto3
from botocore.exceptions import ClientError
import pickle
import os
import time

bucket_name = "aaaaaaaaaaaaaaaaaaa"

my_api_key = "XXXXXXXXXXXXXXXXXXXXXXXXX"
my_api_secret = "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"

search_query = "#btc -filter:retweets"

max_id = 0

while True:
    try:
        auth = tw.OAuthHandler(my_api_key, my_api_secret)
        api = tw.API(auth, wait_on_rate_limit=True)

        # get tweets from the API
        tweets = tw.Cursor(api.search_tweets,
                    q=search_query,
                    since_id=max_id,
                    lang="en").items(1000)

        tweets_copy = []
        for tweet in tweets:
            tweets_copy.append(tweet._json)

        for twt in tweets_copy:
            #print(json.dumps(twt))
            if twt['id'] > max_id:
                max_id = twt['id']

        print("Total Tweets fetched:" + str(len(tweets_copy)) + ", max_id=" + str(max_id))

        fn = "./tweets/" + str(max_id)
        if os.path.exists(fn):
            print('File ' + fn + ' exists. Not uploading to s3')
            continue

        with open(fn, 'wb') as f:
            pickle.dump(tweets_copy, f)

        object_name = "tweets/btc/" + str(max_id)
        s3_client = boto3.client("s3")
        try:
            resp = s3_client.upload_file(fn, bucket_name, object_name)
        except ClientError as ce:
            print(str(ce))

        print('Sleeping for 600 seconds..')
        time.sleep(600)
    except Exception as ex:
        print('Caught ' + ex + '. Sleeping 600 seconds')
        time.sleep(600)

os._exit(0)
