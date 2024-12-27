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


# def get_page_content(title):
#     """获取指定标题的Wikipedia页面内容"""
#     try:
#         page = wikipedia.page(title)
#         page_data = {
#             'title': page.title,
#             'content': page.content,
#             'links': page.links,
#             'references': page.references,
#             'categories': page.categories  # 直接使用所有类别，不进行过滤
#         }
#
#         print(f"Title: {page_data['title']}")
#         print(f"Number of categories: {len(page_data['categories'])}")
#         print(f"Number of links: {len(page_data['links'])}")
#         print(f"First 5 links: {page_data['links'][:5]}")
#         print(f"Categories: {page_data['categories']}")  # 打印类别
#         print("-" * 50)
#
#         return page_data
#     except (wikipedia.exceptions.DisambiguationError, wikipedia.exceptions.PageError):
#         return None


def get_page_content(title):
    """获取指定标题的Wikipedia页面内容，包括图像链接"""
    try:
        page = wikipedia.page(title, auto_suggest=False)
        page_data = {
            'title': page.title,
            'content': page.content,
            'links': page.links,
            'categories': page.categories,
            'images': page.images  # 添加图像链接
        }
        return page_data
    except (wikipedia.exceptions.DisambiguationError, wikipedia.exceptions.PageError, KeyError) as e:
        print(f"Error fetching page '{title}': {str(e)}")
        return None


def build_wiki_subgraph(center_page, G, node_id_map, current_id, max_order=5):
    """构建以给定页面为中心的5阶邻居子图"""
    queue = deque([(center_page, 0)])
    visited = set()

    while queue:
        current_page, order = queue.popleft()
        if current_page in visited or order > max_order:
            continue

        visited.add(current_page)
        page_data = get_page_content(current_page)
        if not page_data:
            continue

        if current_page not in node_id_map:
            node_id_map[current_page] = current_id
            current_id += 1

        node_id = node_id_map[current_page]
        if node_id not in G:
            # 确保不会重复添加 'title' 属性
            if 'title' not in page_data:
                page_data['title'] = current_page
            G.add_node(node_id, **page_data)

        if order < max_order:
            for link in page_data['links']:
                if link not in node_id_map:
                    node_id_map[link] = current_id
                    current_id += 1
                link_id = node_id_map[link]
                if link_id not in G:
                    G.add_node(link_id, title=link)
                G.add_edge(node_id, link_id)

                if link not in visited:
                    queue.append((link, order + 1))

    return G, current_id




def build_wiki_graph(num_center_pages=100, max_order=5, verbose=False):
    """构建Wikipedia图"""
    G = nx.Graph()
    center_pages = set()
    node_id_map = {}
    current_id = 0

    while len(center_pages) < num_center_pages:
        center_page = get_random_wiki_page()
        if center_page and center_page not in center_pages:
            center_pages.add(center_page)
            G, current_id = build_wiki_subgraph(center_page, G, node_id_map, current_id, max_order)

            if verbose:
                print(f"Added subgraph centered at: {center_page}")
                print(f"Current graph size: Nodes={G.number_of_nodes()}, Edges={G.number_of_edges()}")

    return G, node_id_map



def save_graph_and_labels(G, node_id_map, filename_prefix):
    """保存图结构、节点标签和节点信息，包括图像链接"""
    # 保存图结构
    nx.write_gexf(G, f"{filename_prefix}_graph.gexf")

    # 保存节点标签（使用类别作为标签）
    labels = {node: data.get('categories', []) for node, data in G.nodes(data=True)}
    with open(f"{filename_prefix}_labels.json", 'w') as f:
        json.dump(labels, f)

    # 保存额外的节点信息，包括图像链接和原始标题
    node_info = {node: {
        'id': node,
        'title': data.get('title', ''),
        'categories': data.get('categories', []),
        'images': data.get('images', [])
    } for node, data in G.nodes(data=True)}
    with open(f"{filename_prefix}_node_info.json", 'w') as f:
        json.dump(node_info, f)

    # 保存节点ID映射
    with open(f"{filename_prefix}_node_id_map.json", 'w') as f:
        json.dump(node_id_map, f)


def main():
    parser = argparse.ArgumentParser(description="Build a Wikipedia graph")
    parser.add_argument("--num_center_pages", type=int, default=1, help="Number of center pages")
    parser.add_argument("--max_order", type=int, default=5, help="Maximum order for each subgraph")
    parser.add_argument("--verbose", action="store_true", help="Print detailed information")
    parser.add_argument("--output", type=str, default="wiki_graph", help="Output file prefix")
    args = parser.parse_args()

    # 构建图
    wiki_graph, node_id_map = build_wiki_graph(args.num_center_pages, args.max_order, args.verbose)

    # 打印基本信息
    print(f"Final graph size: Nodes={wiki_graph.number_of_nodes()}, Edges={wiki_graph.number_of_edges()}")

    # 保存图和标签
    save_graph_and_labels(wiki_graph, node_id_map, args.output)

    print(f"Graph saved to {args.output}_graph.gexf")
    print(f"Labels saved to {args.output}_labels.json")
    print(f"Node info saved to {args.output}_node_info.json")
    print(f"Node ID map saved to {args.output}_node_id_map.json")


if __name__ == "__main__":
    main()