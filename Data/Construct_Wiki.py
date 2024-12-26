import wikipedia
import networkx as nx
import random
import json
import argparse
import requests
from datetime import datetime, timedelta



def get_page_views(title, days=30):
    """
    获取指定页面最近30天的总浏览量
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    # 构造API URL
    url = f"https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/all-access/all-agents/{title}/daily/{start_date.strftime('%Y%m%d')}/{end_date.strftime('%Y%m%d')}"

    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        total_views = sum(item['views'] for item in data['items'])
        return total_views
    else:
        print(f"Error fetching page views for {title}: {response.status_code}")
        return 0



def get_random_wiki_page():
    """获取一个随机的Wikipedia页面"""
    try:
        return wikipedia.random(1)
    except wikipedia.exceptions.WikipediaException:
        return None


def get_page_content(title):
    """获取指定标题的Wikipedia页面内容"""
    try:
        page = wikipedia.page(title)
        page_views = get_page_views(page.title.replace(" ", "_"))
        page_data = {
            'title': page.title,
            'content': page.content,
            'links': page.links,
            'categories': page.categories,
            'views': page_views
        }

        # 打印页面信息
        print(f"Title: {page_data['title']}")
        print(f"Content (first 100 characters): {page_data['content'][:100]}...")
        print(f"Number of links: {len(page_data['links'])}")
        print(f"First 5 links: {page_data['links'][:5]}")
        print(f"Categories: {page_data['categories']}")
        print(f"Page views (last 30 days): {page_data['views']}")
        print("-" * 50)  # 分隔线

        return page_data
    except (wikipedia.exceptions.DisambiguationError, wikipedia.exceptions.PageError):
        return None


def classify_page_views(views):
    """
    将页面浏览量划分为10个层级
    """
    if views == 0:
        return 0
    elif views < 10:
        return 1
    elif views < 100:
        return 2
    elif views < 1000:
        return 3
    elif views < 10000:
        return 4
    elif views < 100000:
        return 5
    elif views < 1000000:
        return 6
    elif views < 10000000:
        return 7
    elif views < 100000000:
        return 8
    else:
        return 9



def build_wiki_graph(num_pages=10, verbose=False):
    """构建Wikipedia页面图"""
    G = nx.DiGraph()
    pages = {}

    while len(pages) < num_pages:
        title = get_random_wiki_page()
        if title and title not in pages:
            page_data = get_page_content(title)
            if page_data:
                pages[title] = page_data
                view_class = classify_page_views(page_data['views'])
                G.add_node(title, content=page_data['content'],
                           categories=page_data['categories'],
                           views=page_data['views'],
                           view_class=view_class)

                if verbose:
                    print(f"Added page: {title}")
                    print(f"View class: {view_class}")

                # 添加链接
                for link in page_data['links']:
                    if link in pages:
                        G.add_edge(title, link)

    return G


def save_graph_and_labels(G, filename_prefix):
    # 保存图结构
    nx.write_gexf(G, f"{filename_prefix}_graph.gexf")

    # 保存节点标签（使用浏览量分类作为标签）
    labels = {node: data['view_class'] for node, data in G.nodes(data=True)}
    with open(f"{filename_prefix}_labels.json", 'w') as f:
        json.dump(labels, f)

    # 保存额外的节点信息
    node_info = {node: {'views': data['views'], 'categories': data['categories']}
                 for node, data in G.nodes(data=True)}
    with open(f"{filename_prefix}_node_info.json", 'w') as f:
        json.dump(node_info, f)



def main():
    parser = argparse.ArgumentParser(description="Build a Wikipedia graph")
    parser.add_argument("--num_pages", type=int, default=10, help="Number of pages to fetch")
    parser.add_argument("--verbose", action="store_true", help="Print detailed information")
    parser.add_argument("--output", type=str, default="wiki_data", help="Output file prefix")
    args = parser.parse_args()

    # 构建包含指定数量页面的图
    wiki_graph = build_wiki_graph(args.num_pages, args.verbose)

    # 打印一些基本信息
    print(f"Number of nodes: {wiki_graph.number_of_nodes()}")
    print(f"Number of edges: {wiki_graph.number_of_edges()}")

    # 保存图和标签
    save_graph_and_labels(wiki_graph, args.output)

    print(f"Graph saved to {args.output}_graph.gexf")
    print(f"Labels saved to {args.output}_labels.json")
    print(f"Node info saved to {args.output}_node_info.json")


if __name__ == "__main__":
    main()
