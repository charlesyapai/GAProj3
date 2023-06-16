import praw
import pandas as pd
reddit = praw.Reddit(client_id ='zWndV5hUJ8XZYshw1V9axw',
                     client_secret ='4ovj54M6J9s4yS6alG7c58jKOa3TQA',
                     user_agent ='my user agent')
subs = ['bipolar', 'schizophrenia']
submissions_types = ['hot', 'controversial', 'new', 'rising', 'top']
for x in subs:
    posts = []
    subreddit = reddit.subreddit(x)

    for submission_type in submissions_types:
        submission_generator = getattr(subreddit, submission_type)(limit=1000)
        posts.extend([[submission.id, submission.title, submission.selftext, submission.score] for submission in submission_generator])
        print(f'{submission_type.capitalize()} posts from {x} subreddit scraped')

    df = pd.DataFrame(posts, columns=['id', 'title', 'text', 'score'])
    df.to_csv(f'{x}.csv')