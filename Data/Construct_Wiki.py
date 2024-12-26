import wikipedia
import networkx as nx
import random
import json
import argparse
from collections import deque



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


def build_wiki_subgraph(center_page, G, max_depth=3):
    """构建以给定页面为中心的子图"""
    queue = deque([(center_page, 0)])
    visited = set()

    while queue:
        current_page, depth = queue.popleft()
        if current_page in visited or depth > max_depth:
            continue

        visited.add(current_page)
        page_data = get_page_content(current_page)
        if not page_data:
            continue

        if current_page not in G:
            G.add_node(current_page, **page_data)

        if depth < max_depth:
            for ref in page_data['references']:
                if ref not in visited:
                    queue.append((ref, depth + 1))
                if ref not in G:
                    G.add_node(ref)
                G.add_edge(current_page, ref)

    return G



def build_wiki_graph(num_center_pages=100, max_depth=3, verbose=False):
    """构建Wikipedia图"""
    G = nx.Graph()
    center_pages = set()

    while len(center_pages) < num_center_pages:
        center_page = wikipedia.random(1)
        if center_page not in center_pages:
            center_pages.add(center_page)
            G = build_wiki_subgraph(center_page, G, max_depth)

            if verbose:
                print(f"Added subgraph centered at: {center_page}")
                print(f"Current graph size: Nodes={G.number_of_nodes()}, Edges={G.number_of_edges()}")

    return G


def save_graph_and_labels(G, filename_prefix):
    """保存图结构和节点标签"""
    # 保存图结构
    nx.write_gexf(G, f"{filename_prefix}_graph.gexf")

    # 保存节点标签（使用类别作为标签）
    labels = {node: data.get('categories', []) for node, data in G.nodes(data=True)}
    with open(f"{filename_prefix}_labels.json", 'w') as f:
        json.dump(labels, f)

    # 保存额外的节点信息
    node_info = {node: {
        'title': data.get('title', ''),
        'categories': data.get('categories', [])
    } for node, data in G.nodes(data=True)}
    with open(f"{filename_prefix}_node_info.json", 'w') as f:
        json.dump(node_info, f)


def main():
    parser = argparse.ArgumentParser(description="Build a Wikipedia graph")
    parser.add_argument("--num_center_pages", type=int, default=100, help="Number of center pages")
    parser.add_argument("--max_depth", type=int, default=3, help="Maximum depth for each subgraph")
    parser.add_argument("--verbose", action="store_true", help="Print detailed information")
    parser.add_argument("--output", type=str, default="wiki_graph", help="Output file prefix")
    args = parser.parse_args()

    # 构建图
    wiki_graph = build_wiki_graph(args.num_center_pages, args.max_depth, args.verbose)

    # 打印基本信息
    print(f"Final graph size: Nodes={wiki_graph.number_of_nodes()}, Edges={wiki_graph.number_of_edges()}")

    # 保存图和标签
    save_graph_and_labels(wiki_graph, args.output)

    print(f"Graph saved to {args.output}_graph.gexf")
    print(f"Labels saved to {args.output}_labels.json")
    print(f"Node info saved to {args.output}_node_info.json")


if __name__ == "__main__":
    main()