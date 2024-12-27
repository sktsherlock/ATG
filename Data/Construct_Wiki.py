import wikipedia
import networkx as nx
import random
import json
import argparse
from collections import deque
from tqdm import tqdm


def get_page_content(title):
    """获取指定标题的Wikipedia页面内容，包括图像链接"""
    try:
        page = wikipedia.page(title, auto_suggest=False)
        images = [img for img in page.images if img.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))]
        if not images:  # 如果没有有效的图像链接，返回 None
            # print(f"Skipping page '{title}' due to lack of images.")
            return None
        page_data = {
            'title': page.title,
            'pageid': page.pageid,
            'content': page.content,
            'links': page.links,
            'categories': page.categories,
            'images': images
        }
        return page_data
    except (wikipedia.exceptions.DisambiguationError, wikipedia.exceptions.PageError, KeyError) as e:
        print(f"Error fetching page '{title}': {str(e)}")
        return None


def build_wiki_subgraph(center_page, G, node_id_map, current_id, max_order=5, max_links=50):
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

        # 检查是否有图像
        if not page_data.get('images'):
            # print(f"Skipping page '{current_page}' due to lack of images.")
            continue

        if current_page not in node_id_map:
            node_id_map[current_page] = current_id
            current_id += 1

        node_id = node_id_map[current_page]
        if node_id not in G:
            if 'title' not in page_data:
                page_data['title'] = current_page
            G.add_node(node_id, **page_data)
            print(f"Added page: '{current_page}' with node ID: {node_id}")

        if order < max_order:
            # 预处理链接：如果链接数量超过 max_links，随机采样
            links = page_data['links']
            if len(links) > max_links:
                print(f"Page '{current_page}' has {len(links)} links. Sampling {max_links} links.")
                links = random.sample(links, max_links)

            for link in links:
                link_data = get_page_content(link)
                if not link_data or not link_data.get('images'):
                    # print(f"Skipping linked page '{link}' due to lack of images or content.")
                    continue

                if link not in node_id_map:
                    node_id_map[link] = current_id
                    current_id += 1
                link_id = node_id_map[link]
                if link_id not in G:
                    G.add_node(link_id, **link_data)
                    print(f"Added linked page: '{link}' with node ID: {link_id}")
                G.add_edge(node_id, link_id)

                if link not in visited:
                    queue.append((link, order + 1))

    return G, current_id, list(visited)


def build_wiki_graph(initial_page, num_iterations=5, max_order=5, verbose=False):
    """构建Wikipedia图"""
    G = nx.Graph()
    node_id_map = {}
    current_id = 0
    center_page = initial_page

    with tqdm(total=num_iterations, desc="Building Wiki Graph") as pbar:
        for i in range(num_iterations):
            print(f"\nIteration {i + 1}: Building subgraph for center page: '{center_page}'")
            G, current_id, visited_pages = build_wiki_subgraph(center_page, G, node_id_map, current_id, max_order)

            if verbose:
                print(f"Current graph size: Nodes={G.number_of_nodes()}, Edges={G.number_of_edges()}")

            # 选择下一个中心页面
            if i < num_iterations - 1:  # 如果不是最后一次迭代
                center_page = random.choice(visited_pages)
                print(f"Next center page: '{center_page}'")

            pbar.update(1)

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
    parser.add_argument("--initial_page", type=str, required=True, help="Initial center page title")
    parser.add_argument("--num_iterations", type=int, default=5, help="Number of iterations")
    parser.add_argument("--max_order", type=int, default=2, help="Maximum order for each subgraph")
    parser.add_argument("--verbose", action="store_true", help="Print detailed information")
    parser.add_argument("--output", type=str, default="wiki_graph", help="Output file prefix")
    args = parser.parse_args()

    # 构建图
    wiki_graph, node_id_map = build_wiki_graph(args.initial_page, args.num_iterations, args.max_order, args.verbose)

    # 打印基本信息
    print(f"\nFinal graph size: Nodes={wiki_graph.number_of_nodes()}, Edges={wiki_graph.number_of_edges()}")

    # 保存图和标签
    save_graph_and_labels(wiki_graph, node_id_map, args.output)

    print(f"Graph saved to {args.output}_graph.gexf")
    print(f"Labels saved to {args.output}_labels.json")
    print(f"Node info saved to {args.output}_node_info.json")
    print(f"Node ID map saved to {args.output}_node_id_map.json")


if __name__ == "__main__":
    main()
