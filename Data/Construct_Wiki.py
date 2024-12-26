import wikipedia
import networkx as nx
import random
import argparse


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
            'categories': page.categories  # 添加这一行来获取页面类别
        }

        # 打印页面信息
        print(f"Title: {page_data['title']}")
        print(f"Content (first 100 characters): {page_data['content'][:100]}...")
        print(f"Number of links: {len(page_data['links'])}")
        print(f"First 5 links: {page_data['links'][:5]}")
        print(f"Categories: {page_data['categories']}")  # 打印类别
        print("-" * 50)  # 分隔线

        return page_data
    except (wikipedia.exceptions.DisambiguationError, wikipedia.exceptions.PageError):
        return None


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
                G.add_node(title, content=page_data['content'])

                if verbose:
                    print(f"Added page: {title}")

                # 添加链接
                for link in page_data['links']:
                    if link in pages:
                        G.add_edge(title, link)

    return G


def main():
    parser = argparse.ArgumentParser(description="Build a Wikipedia graph")
    parser.add_argument("--num_pages", type=int, default=10, help="Number of pages to fetch")
    parser.add_argument("--verbose", action="store_true", help="Print detailed information")
    args = parser.parse_args()

    # 构建包含指定数量页面的图
    wiki_graph = build_wiki_graph(args.num_pages, args.verbose)

    # 打印一些基本信息
    print(f"Number of nodes: {wiki_graph.number_of_nodes()}")
    print(f"Number of edges: {wiki_graph.number_of_edges()}")

    # 保存图
    nx.write_gexf(wiki_graph, "wiki_graph.gexf")


if __name__ == "__main__":
    main()
