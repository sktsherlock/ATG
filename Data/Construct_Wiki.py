import wikipedia
import networkx as nx
import random
import json
import argparse
from queue import Queue
from datetime import datetime, timedelta


def get_science_tech_categories():
    """获取科学与技术相关的顶级类别"""
    return [
        "Category:Science",
        "Category:Technology",
        "Category:Engineering",
        "Category:Mathematics",
        "Category:Computer science",
        "Category:Physics",
        "Category:Chemistry",
        "Category:Biology"
    ]


def get_subcategories(category, depth=1):
    """获取给定类别的子类别"""
    if depth == 0:
        return []
    try:
        page = wikipedia.page(category)
        subcats = [cat for cat in page.categories if cat.startswith("Category:")]
        result = subcats[:]
        for subcat in subcats:
            result.extend(get_subcategories(subcat, depth - 1))
        return result
    except:
        return []


def get_pages_in_category(category):
    """获取给定类别中的页面"""
    try:
        return wikipedia.page(category).links
    except:
        return []



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
        page_data = {
            'title': page.title,
            'content': page.content,
            'links': page.links,
            'categories': page.categories  # 直接使用所有类别，不进行过滤
        }

        print(f"Title: {page_data['title']}")
        print(f"Number of categories: {len(page_data['categories'])}")
        print(f"Number of links: {len(page_data['links'])}")
        print(f"First 5 links: {page_data['links'][:5]}")
        print(f"Categories: {page_data['categories']}")  # 打印类别
        print("-" * 50)

        return page_data
    except (wikipedia.exceptions.DisambiguationError, wikipedia.exceptions.PageError):
        return None


def build_wiki_graph(num_pages=1000, verbose=False):
    """构建Wikipedia科学技术相关页面图"""
    G = nx.DiGraph()
    pages = {}
    categories_to_explore = Queue()
    for cat in get_science_tech_categories():
        categories_to_explore.put(cat)

    while len(pages) < num_pages and not categories_to_explore.empty():
        current_category = categories_to_explore.get()

        # 获取子类别并加入队列
        subcats = get_subcategories(current_category, depth=1)
        for subcat in subcats:
            categories_to_explore.put(subcat)

        # 获取当前类别中的页面
        category_pages = get_pages_in_category(current_category)
        for title in category_pages:
            if title not in pages and len(pages) < num_pages:
                page_data = get_page_content(title)
                if page_data:
                    pages[title] = page_data
                    G.add_node(title, content=page_data['content'],
                               categories=page_data['categories'])

                    if verbose:
                        print(f"Added page: {title}")
                        print(f"Categories: {page_data['categories']}")

                    # 添加链接
                    for link in page_data['links']:
                        if link in pages:
                            G.add_edge(title, link)

    return G


def save_graph_and_labels(G, filename_prefix):
    # 保存图结构
    nx.write_gexf(G, f"{filename_prefix}_graph.gexf")

    # 保存节点标签（使用所有类别作为标签）
    labels = {node: data['categories'] for node, data in G.nodes(data=True)}
    with open(f"{filename_prefix}_labels.json", 'w') as f:
        json.dump(labels, f)

    # 保存额外的节点信息
    node_info = {node: {'categories': data['categories']}
                 for node, data in G.nodes(data=True)}
    with open(f"{filename_prefix}_node_info.json", 'w') as f:
        json.dump(node_info, f)


def main():
    parser = argparse.ArgumentParser(description="Build a Wikipedia graph for science and technology")
    parser.add_argument("--num_pages", type=int, default=10, help="Number of pages to fetch")
    parser.add_argument("--verbose", action="store_true", help="Print detailed information")
    parser.add_argument("--output", type=str, default="wiki_science_tech", help="Output file prefix")
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
