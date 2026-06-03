import os
import googleapiclient.discovery
from dotenv import load_dotenv # you need a .env file with your YouTube API key


# initiliaze API key and YouTube API client 
load_dotenv() 
api_key = os.getenv("YOUTUBE_API_KEY")
if not api_key:
    raise ValueError("YOUTUBE_API_KEY not found. Please check the .env File!")

youtube = googleapiclient.discovery.build('youtube', 'v3', developerKey = api_key)


# Function for retrieving video IDs with paging
def get_video_ids(query, max_videos = 1000): 
    # max_videos is the maximum number of video IDs to retrieve, default is 1000, but you can adjust it as needed, 
    # just don't set it too high, otherwise you might hit the API limit 
    video_ids = []
    request = youtube.search().list(
        part = 'id',
        q = query,
        type = 'video',
        maxResults = 50  # Max per Request, you can go lower if you want, but not higher (API restrictions)
    )
    
    while request and len(video_ids) < max_videos:
        response = request.execute()
        
        for item in response['items']:
            vid = item['id']['videoId']
            if vid not in video_ids:
                video_ids.append(vid)
                if len(video_ids) >= max_videos:
                    break
        
        request = youtube.search().list_next(request, response)  # call next page
    
    return video_ids

# collect video IDs for different topics, you can adjust the topics as needed, just comment out the ones you don't need (only one topic at a time)
# Funny Videos
topics = ["Viral Video", "Challenge Video", "Funny Clip", "Meme Compilation", "TikTok Compilation", "Funny Shorts"]
# Tech
topics = ["Gadget Review", "Smartphone Review", "Software Tutorial", "Gaming Let's Play", "Game Walkthrough", "Game Tutorial", "Tech News", "Startup Presentation", "Technology Updates"]
# Education
topics = ["Tutorial Video", "How-to Video", "Instruction Video", "Physics Experiment", "Biology Channel", "Technology Knowledge", "Learn Languages", "Language Learning", "Vocabulary Tutorial", "History Documentary", "Historical Documentary", "Documentary Channel"]
# Entertainment
topics = ["Music Video", "Pop Song Video", "Top Charts 2025", "Movie Trailer", "TV Series Trailer", "Blockbuster Trailer", "Comedy Sketch", "Stand-up Comedy", "Funny Videos", "Gaming Let's Play", "Funny Gaming Moments", "Game Clips"]
# Sport 
topics = ["Sports Highlight", "Sports Live Clip", "Goal Video", "Training Tutorial", "Fitness Exercise", "Workout Video", "Sports Analysis", "Game Commentary", "Tactics Video"]
# Lifestyle
topics = ["Daily Vlog", "YouTuber Vlog", "Lifestyle Vlog", "Travel Adventure", "Travel Vlog", "Backpacking Video", "Fitness Training", "Nutrition Tips", "Health Video", "Fashion Tips", "Beauty Tutorial", "Fashion Haul"]
# News
topics = ["News Clip", "Breaking News", "Daily News", "Talk Show Highlights", "Political Debate", "Discussion Show", "Political Analysis", "Political Commentary", "Opinion Video", "NGO Video", "Activism Channel", "Charity Video"]


all_video_ids = []

for topic in topics:
    print(f"collect video-IDs from topic: {topic} ...")
    all_video_ids.extend(get_video_ids(topic, max_videos = 100))  #  max videos per topic, you can adjust the number, but don't set it too high, otherwise you might hit the API limit

# Remove duplicates
unique_video_ids = list(set(all_video_ids))


# Save the video IDs to a text file
output_file = 'video_ids.txt'
with open(output_file, 'w', encoding='utf-8') as f:
    for vid in unique_video_ids:
        f.write(f"{vid}\n")

print(f"{len(unique_video_ids)} video-IDs have been succesfully saved in '{output_file}'.")
