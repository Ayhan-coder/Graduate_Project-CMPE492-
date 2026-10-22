import json
import yt_dlp

def search_domain_videos():
    queries = {
        "kvkk": "KVKK temel farkındalık eğitimi",
        "isg": "temel iş sağlığı ve güvenliği eğitimi"
    }
    
    results = {}
    ydl_opts = {
        'quiet': True,
        'extract_flat': True,
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        for domain, q in queries.items():
            print(f"Searching for {domain}: {q}")
            res = ydl.extract_info(f"ytsearch4:{q}", download=False)
            items = []
            for entry in res.get('entries', []):
                items.append({
                    "id": entry.get("id"),
                    "title": entry.get("title"),
                    "duration": entry.get("duration"),
                    "uploader": entry.get("uploader"),
                    "url": f"https://www.youtube.com/watch?v={entry.get('id')}"
                })
            results[domain] = items

    print(json.dumps(results, ensure_ascii=False, indent=2))
    with open("experiments/data/searched_domain_videos.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    search_domain_videos()
