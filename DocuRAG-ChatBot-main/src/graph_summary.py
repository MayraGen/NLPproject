import matplotlib.pyplot as plt
"""
The functions `generate_keyword_table` and `generate_bar_chart` help generate a Markdown-formatted
keyword frequency table and a horizontal bar chart of keyword frequencies, respectively.

:param keywords: The `keywords` parameter is a list containing the keywords for which you want to
generate the keyword frequency table and bar chart. Each element in the list represents a keyword
:type keywords: list
:param counts: The `counts` parameter in the functions `generate_keyword_table` and
`generate_bar_chart` represents the frequency or count of each keyword in a list of keywords. It is
used to generate a keyword frequency table in Markdown format and a horizontal bar chart showing
the frequencies of the keywords
:type counts: list
:return: The `generate_bar_chart` function returns the file path where the generated horizontal bar
chart of keyword frequencies is saved.
"""

import numpy as np
import os
import matplotlib.pyplot as plt
from wordcloud import WordCloud

def generate_keyword_table(keywords: list, counts: list) -> str:
    """Generate Markdown-formatted keyword frequency table."""
    if not keywords or not counts:
        return "No keywords found"
    
    table = "| Keyword | Count |\n|---------|-------|\n"
    for kw, cnt in zip(keywords, counts):
        table += f"| {kw} | {cnt} |\n"
    return table

def generate_bar_chart(keywords: list, counts: list) -> str:
    """Generate horizontal bar chart of keyword frequencies."""
    if not keywords or not counts:
        return None
    
    os.makedirs("outputs", exist_ok=True)
    
    sorted_indices = np.argsort(counts)[::-1]
    keywords = [keywords[i] for i in sorted_indices]
    counts = [counts[i] for i in sorted_indices]
    
    fig_height = max(4, len(keywords) * 0.5)
    plt.figure(figsize=(10, fig_height))
    
    y_pos = np.arange(len(keywords))
    plt.barh(y_pos, counts, align='center', color='skyblue')
    plt.yticks(y_pos, labels=keywords)
    plt.xlabel('Frequency', fontsize=12)
    plt.title('Top Keyword Frequencies', fontsize=14, pad=20)
    
    plt.gca().invert_yaxis()
    plt.tight_layout()
    
    chart_path = "outputs/keywords_bar.png"
    plt.savefig(chart_path, bbox_inches='tight')
    plt.close()
    
    return chart_path

def generate_pie_chart(keywords: list, counts: list) -> str:
    """Generate a pie chart for keyword distribution."""
    if not keywords or not counts:
        return None

    os.makedirs("outputs", exist_ok=True)

    plt.figure(figsize=(8, 8))
    plt.pie(counts, labels=keywords, autopct='%1.1f%%', colors=plt.cm.Paired.colors, startangle=140)
    plt.axis('equal')  
    plt.title('Keyword Distribution', fontsize=14)

    pie_chart_path = "outputs/keywords_pie.png"
    plt.savefig(pie_chart_path, bbox_inches='tight')
    plt.close()
    
    return pie_chart_path

def generate_word_cloud(keywords: list, counts: list) -> str:
    """Generate a word cloud image based on keyword frequencies."""
    if not keywords or not counts:
        return None
    
    os.makedirs("outputs", exist_ok=True)
    
    word_freq = dict(zip(keywords, counts))
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freq)
    
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.title("Word Cloud of Keywords", fontsize=14)
    
    wordcloud_path = "outputs/keywords_wordcloud.png"
    plt.savefig(wordcloud_path, bbox_inches='tight')
    plt.close()
    
    return wordcloud_path


