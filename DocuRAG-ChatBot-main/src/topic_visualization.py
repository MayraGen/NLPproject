import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from src.document_processing import chunk_text  # Make sure this import is correct

def generate_topic_visualization(text):
    """Generate advanced topic relationship visualization
    Returns:
        tuple: (network_fig, projection_fig, topic_terms)
    """
    try:
        # Process text
        chunks = chunk_text(text)[:100]  # Use first 100 chunks
        if not chunks:
            return None, None, []
            
        vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        X = vectorizer.fit_transform(chunks)
        
        # Train LDA model
        lda = LatentDirichletAllocation(n_components=5, random_state=42)
        lda.fit(X)
        
        # Create separate figures for Gradio components
        network_fig = plt.figure(figsize=(10, 6))
        projection_fig = plt.figure(figsize=(8, 6))
        topic_terms = []

        # 1. Topic-Term Network
        ax1 = network_fig.add_subplot(111)
        G = nx.Graph()
        
        # Add nodes and edges
        for topic_idx, topic in enumerate(lda.components_):
            top_terms = [vectorizer.get_feature_names_out()[i] for i in topic.argsort()[-5:]]
            topic_terms.append((f"Topic {topic_idx+1}", ", ".join(top_terms)))
            G.add_node(f"Topic {topic_idx+1}", type='topic')
            for term in top_terms:
                G.add_node(term, type='term')
                G.add_edge(f"Topic {topic_idx+1}", term, weight=topic[vectorizer.vocabulary_[term]])

        # Network visualization
        pos = nx.spring_layout(G)
        nx.draw_networkx_nodes(G, pos, nodelist=[n for n in G.nodes if G.nodes[n]['type']=='topic'], 
                            node_color='red', node_size=800, ax=ax1)
        nx.draw_networkx_nodes(G, pos, nodelist=[n for n in G.nodes if G.nodes[n]['type']=='term'], 
                            node_color='skyblue', node_size=400, ax=ax1)
        nx.draw_networkx_edges(G, pos, edge_color='gray', alpha=0.4, ax=ax1)
        nx.draw_networkx_labels(G, pos, ax=ax1)
        ax1.set_title("Topic-Term Relationships")
        plt.close(network_fig)  # Close figure to prevent memory leaks

        # 2. Topic Projection
        ax2 = projection_fig.add_subplot(111)
        topic_vectors = lda.transform(X)
        # To this (add perplexity adjustment):
        n_samples = len(chunks)
        tsne = TSNE(
            n_components=2,
            random_state=42,
            perplexity=min(30, (n_samples - 1) // 3)  # Auto-adjust perplexity
        )
        projected = tsne.fit_transform(topic_vectors)
        
        # Plot clusters
        for i in range(lda.n_components):
            ax2.scatter(projected[topic_vectors.argmax(axis=1) == i, 0],
                    projected[topic_vectors.argmax(axis=1) == i, 1],
                    label=f"Topic {i+1}", alpha=0.6)
        ax2.set_xlabel("TSNE-1")
        ax2.set_ylabel("TSNE-2")
        ax2.set_title("Topic Projection Space")
        ax2.legend()
        plt.close(projection_fig)  # Close figure to prevent memory leaks

        return network_fig, projection_fig, topic_terms

    except Exception as e:
        print(f"Visualization error: {e}")
        return None, None, []







# from sklearn.decomposition import NMF
# from sklearn.feature_extraction.text import CountVectorizer

# # Example: Your cleaned text
# documents = ["This is the first document.", 
#              "This is the second text about science.", 
#              "Another text on machine learning.", 
#              "Deep learning and AI models are fascinating."]

# # Convert text into a bag-of-words matrix
# vectorizer = CountVectorizer(stop_words='english')
# X = vectorizer.fit_transform(documents)

# # Apply topic modeling (NMF as an example)
# nmf_model = NMF(n_components=2, random_state=42)
# W = nmf_model.fit_transform(X)
# H = nmf_model.components_

# # Prepare visualization
# topic_data = tw.prepare(W, H, vectorizer.get_feature_names_out())

# # Launch interactive visualization
# tw.visualize(topic_data)