import os
from typing import Generator
import googleapiclient.discovery
from googleapiclient.errors import HttpError
from dotenv import load_dotenv
from langdetect import detect, LangDetectException

load_dotenv()

class YouTubeScraper:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("YOUTUBE_API_KEY")
        if not self.api_key:
            raise ValueError("YOUTUBE_API_KEY not found. Please check your .env file!")
        
        self.youtube = googleapiclient.discovery.build("youtube", "v3", developerKey=self.api_key)

    @staticmethod
    def is_english(text: str) -> bool:
        # filter out non-English comments using langdetect
        try:
            return detect(text) == "en"
        except LangDetectException:
            return False

    def get_video_ids_for_topic(self, topic: str, max_videos: int = 50) -> list[str]:
        # Retrieves video IDs for a given topic using the YouTube Data API.
        video_ids = []
        request = self.youtube.search().list(
            part="id",
            q=topic,
            type="video",
            maxResults=min(50, max_videos)
        )
        
        while request and len(video_ids) < max_videos:
            response = request.execute()
            for item in response.get("items", []):
                vid = item["id"]["videoId"]
                if vid not in video_ids:
                    video_ids.append(vid)
                    if len(video_ids) >= max_videos:
                        break
            request = self.youtube.search().list_next(request, response)
            
        return video_ids



    def fetch_comments_for_video(self, video_id: str, max_comments: int = 1000, filter_english: bool = True) -> Generator[dict, None, None]:
        # Fetches comments for a given video ID using the YouTube Data API.
        fetched = 0
        next_page_token = None

        while fetched < max_comments:
            try:
                response = self.youtube.commentThreads().list(
                    part="snippet",
                    videoId=video_id,
                    maxResults=100,
                    pageToken=next_page_token,
                    textFormat="plainText",
                ).execute()
            except HttpError as e:
                print(f"⚠️ Skipping video {video_id} due to API error: {e}")
                break

            for item in response.get("items", []):
                snippet = item["snippet"]["topLevelComment"]["snippet"]
                text = snippet["textDisplay"]

                if filter_english and not self.is_english(text):
                    continue

                yield {
                    "videoId": video_id,
                    "comment": text,
                    "author": snippet.get("authorDisplayName"),
                    "publishedAt": snippet.get("publishedAt"),
                    "likeCount": snippet.get("likeCount", 0),
                    "replyCount": item["snippet"].get("totalReplyCount", 0)
                }
                fetched += 1
                if fetched >= max_comments:
                    break

            next_page_token = response.get("nextPageToken")
            if not next_page_token:
                break