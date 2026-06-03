import os
import json
from pathlib import Path
import googleapiclient.discovery
from googleapiclient.errors import HttpError
from dotenv import load_dotenv
from langdetect import detect, LangDetectException
from collections import Counter

# initialize API key and YouTube API client
load_dotenv()
api_key = os.getenv("YOUTUBE_API_KEY")
if not api_key:
    raise ValueError("YOUTUBE_API_KEY not found. Please check the .env File!")

youtube = googleapiclient.discovery.build("youtube", "v3", developerKey=api_key)

# paths
COMMENTS_FILE = Path("comments.json")
VIDEO_IDS_FILE = Path("video_ids.txt")

# global language stats for developer mode
language_stats = Counter()



def load_comments():
    if COMMENTS_FILE.exists():
        try:
            with open(COMMENTS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            print(f"warning: {COMMENTS_FILE} is empty or invalid. Start with empty list.")
            return []
    return []


def load_video_ids():
    if not VIDEO_IDS_FILE.exists():
        print(f"{VIDEO_IDS_FILE} not found.")
        return []
    with open(VIDEO_IDS_FILE, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def is_english(text: str, dev_mode: bool = False) -> bool:
    """Detects if text is English. In dev_mode counts all detected languages."""
    try:
        lang = detect(text)
        if dev_mode:
            language_stats[lang] += 1
        return lang == "en"
    except LangDetectException:
        if dev_mode:
            language_stats["unknown"] += 1
        return False


def get_comments(video_id, max_comments=1000, dev_mode: bool = False):
    print(f"get comments from video {video_id} ...")
    all_comments = []
    next_page_token = None

    while True:
        try:
            response = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=100,
                pageToken=next_page_token,
                textFormat="plainText",
            ).execute()
        except HttpError as e:
            error_reason = (
                e.error_details[0]["reason"]
                if hasattr(e, "error_details") and e.error_details
                else str(e)
            )
            print(f"skipping video {video_id} due to error: {error_reason}")
            break

        for item in response.get("items", []):
            snippet = item["snippet"]["topLevelComment"]["snippet"]
            text = snippet["textDisplay"]

            if is_english(text, dev_mode=dev_mode):
                all_comments.append({
                    "videoId": video_id,
                    "comment": text,
                    "author": snippet["authorDisplayName"],
                    "authorChannelId": snippet.get("authorChannelId", {}).get("value"),
                    "publishedAt": snippet.get("publishedAt"),
                    "likeCount": snippet.get("likeCount", 0),
                    "replyCount": item["snippet"].get("totalReplyCount", 0),
                })

            if len(all_comments) >= max_comments:
                break

        next_page_token = response.get("nextPageToken")
        if not next_page_token or len(all_comments) >= max_comments:
            break

    return all_comments[:max_comments]


def save_comments(comments):
    with open(COMMENTS_FILE, "w", encoding="utf-8") as f:
        json.dump(comments, f, ensure_ascii=False, indent=2)
    print(f"{len(comments)} English comments have been successfully saved in '{COMMENTS_FILE}'.")


def developer_report():
    """Developer function: show stats of detected languages."""
    print("\n--- Developer Report ---")
    total = sum(language_stats.values())
    for lang, count in language_stats.most_common():
        print(f"{lang}: {count} ({count/total:.2%})")
    print(f"Total detected comments: {total}")
    print("------------------------\n")


def main(dev_mode: bool = False):
    video_ids = load_video_ids()
    if not video_ids:
        print("no Video-IDs found. Please save video IDs to video_ids.txt.")
        return

    comments = load_comments()

    for vid in video_ids:
        new_comments = get_comments(vid, max_comments=1000, dev_mode=dev_mode)
        comments.extend(new_comments)
        print(f"{len(new_comments)} English comments added from video {vid}.")

    save_comments(comments)
    print(f"saved {len(comments)} English comments in total.")

    if dev_mode:
        developer_report()


if __name__ == "__main__":
    # toggle dev_mode=True to enable language statistics
    main(dev_mode=True)
