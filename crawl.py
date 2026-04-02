import openreview
import json
import os
from dotenv import load_dotenv  # 引入 dotenv

# 加载 .env 文件中的环境变量
load_dotenv()

def crawl_papers_by_venue(venue_id, output_file, email, password):
    print(f"Connecting to OpenReview API v2 with account: {email}...")
    
    # 【核心修复】：填入 username 和 password，解除匿名访问限制
    client = openreview.api.OpenReviewClient(
        baseurl='https://api2.openreview.net',
        username=email,
        password=password
    )
    
    all_papers = []
    
    try:
        print(f"Fetching accepted papers for venue: {venue_id}...")
        # 登录后，这里就不会再报 403 Forbidden 了
        notes = client.get_all_notes(content={'venueid': venue_id})
        
        for note in notes:
            paper = {
                "id": note.id,
                "number": note.number,
                "title": note.content.get("title", {}).get("value", ""),
                "authors": note.content.get("authors", {}).get("value", []),
                "abstract": note.content.get("abstract", {}).get("value", ""),
                "keywords": note.content.get("keywords", {}).get("value", []),
                "primary_area": note.content.get("primary_area", {}).get("value", ""),
                "forum_url": f"https://openreview.net/forum?id={note.id}"
            }
            all_papers.append(paper)
            # print(f"正在获取{note}")
            
            # # 每下载 500 篇打印一次进度
            # if len(all_papers) % 500 == 0:
            #     print(f"  ... downloaded {len(all_papers)} papers")
            
        print(f"\nSuccessfully fetched {len(all_papers)} papers.")
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(all_papers, f, ensure_ascii=False, indent=2)
        print(f"Saved to {output_file}")
        
    except openreview.OpenReviewException as e:
        print(f"OpenReview API Error: {e}")

if __name__ == "__main__":
    # 请在这里填入你真实注册的 OpenReview 邮箱和密码
    YOUR_EMAIL = os.getenv("YOUR_EMAIL")
    YOUR_PASSWORD = os.getenv("YOUR_PASSWORD")
    
    # crawl_papers_by_venue(
    #     venue_id="ICLR.cc/2026/Conference", 
    #     output_file="papers/iclr2026_papers.json",
    #     email=YOUR_EMAIL,
    #     password=YOUR_PASSWORD
    # )
    
    # crawl_papers_by_venue(
    #     venue_id="NeurIPS.cc/2025/Conference",
    #     output_file="papers/neurips2025_papers.json",
    #     email=YOUR_EMAIL,
    #     password=YOUR_PASSWORD
    # )

    

    crawl_papers_by_venue(
        venue_id="ICML.cc/2025/Conference",
        output_file="papers/icml2025_papers.json",
        email=YOUR_EMAIL,
        password=YOUR_PASSWORD
    )