from googlesearch import search
import requests
from bs4 import BeautifulSoup
from typing import List, Dict
import time

def google_search(query: str, num_results: int = 5) -> List[Dict]:
    """
    使用Google搜索指定查询词，并返回搜索结果
    
    Args:
        query (str): 搜索查询词
        num_results (int): 需要返回的结果数量，默认为5
        
    Returns:
        List[Dict]: 包含标题、摘要和URL的搜索结果列表
    """
    search_results = []
    
    try:
        # 使用googlesearch-python获取搜索结果URL
        for url in search(query, num_results=num_results, lang="zh"):
            try:
                # 发送请求获取页面内容
                headers = {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
                }
                response = requests.get(url, headers=headers, timeout=5)
                response.raise_for_status()
                
                # 使用BeautifulSoup解析页面
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # 获取页面标题
                title = soup.title.string if soup.title else "无标题"
                
                # 获取页面描述或第一段文字作为摘要
                description = ""
                meta_desc = soup.find("meta", {"name": "description"})
                if meta_desc and meta_desc.get("content"):
                    description = meta_desc["content"]
                else:
                    first_p = soup.find("p")
                    if first_p:
                        description = first_p.text.strip()
                
                # 将结果添加到列表中
                search_results.append({
                    "title": title,
                    "description": description[:300] + "..." if len(description) > 200 else description,
                    "url": url
                })
                
                # 添加延时以避免请求过快
                time.sleep(1)
                
            except Exception as e:
                print(f"获取页面 {url} 时出错: {str(e)}")
                continue
                
    except Exception as e:
        print(f"搜索过程中出错: {str(e)}")
        return []
    
    return search_results

if __name__ == "__main__":
    # 测试代码
    results = google_search("lysteda", 3)
    for result in results:
        print(f"\n标题: {result['title']}")
        print(f"描述: {result['description']}")
        print(f"链接: {result['url']}")
