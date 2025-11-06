import numpy as np 
import streamlit as st
import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx
import plotly.graph_objects as go

st.set_page_config(
    page_title = "Attention Visualizer"
)
    

st.sidebar.image("attentionImage.webp", use_container_width=True)
st.sidebar.title("🔍 Attention Visualizer")
st.sidebar.write(
    """
    Explore how **transformer attention** works!  
    Enter any text, and this app will:
    - Tokenize it  
    - Create embeddings  
    - Generate Query (Q), Key (K), and Value (V) matrices  
    - Compute the attention weights
    - Visualize Using Heatmap
    """
)
st.sidebar.warning("All the Parameters are generated from Normal Distribution. No model training or learned weights are used. It is for Visualising Purpose Only.")
st.sidebar.markdown("---")
st.sidebar.caption("👨‍💻 Built by Shivam Sharma")
st.sidebar.markdown("---")
st.sidebar.warning("I spend a long time developing and optimizing this project to make it useful for everyone. If it helped you or saved your time, consider buying me a coffee via UPI - scan the QR code below. Thanks for your support! ❤")
st.sidebar.image("qr.jpeg", use_container_width=True)

#---------------------------------------------------------

st.header("Visualize Attention like Never Before")
st.success("The attention mechanism in Transformers allows the model to focus on the most relevant parts of an input sequence when generating an output. Instead of processing words sequentially like RNNs, Transformers use self-attention to compute relationships between all words in a sentence in parallel. In self-attention, each word is represented by three vectors — Query (Q), Key (K), and Value (V). The attention score between two words is calculated using the dot product of their Query and Key vectors, followed by a softmax operation to obtain normalized weights. These weights determine how much attention each word should give to others. The final output for each word is a weighted sum of the Value vectors.")
st.markdown("---")
# -------------------------------

text = st.text_input("Enter the text: ", "Example: The cat sats on the mat")

tokens = re.findall(r"\w+|[^\w\s]", text)

def make_unique(tokens):
    counts = {}
    unique_tokens = []
    for t in tokens:
        counts[t] = counts.get(t, 0) + 1
        if counts[t] > 1:
            unique_tokens.append(f"{t}_{counts[t]}")
        else:
            unique_tokens.append(t)
    return unique_tokens


def softmax(x):
    # Subtract max for numerical stability
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

np.random.seed(42)

if text:
    st.write("Tokens:", tokens)
    unique_tokens = make_unique(tokens)


    st.markdown("---")
    with st.expander("📐 Positional Encoding Visualizer"):
        st.subheader("📐 Positional Encoding Visualizer")

        st.write("""
        Transformers don’t have recurrence, so they use **positional encoding** 
        to give each token a sense of order in the sequence.

        The encoding uses sine and cosine functions at different frequencies:
        """)

        with st.expander("📘 Positional Encoding Formula"):
            st.latex(r"""
            PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)
            """)
            st.latex(r"""
            PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)
            """)

        num_positions = st.slider("Number of positions to visualize", 5, 50, 10)
        d_model_pe = st.slider("Encoding dimension (for visualization)", 2, 64, 8)

        pos = np.arange(num_positions)
        dim = np.arange(d_model_pe)

        pe = np.array([
            [
                np.sin(p / (10000 ** (2 * (i//2) / d_model_pe))) if i % 2 == 0 
                else np.cos(p / (10000 ** (2 * (i//2) / d_model_pe))) 
                for i in dim
            ]
            for p in pos
        ])

        pe_df = pd.DataFrame(pe, columns=[f"Dim {i+1}" for i in range(d_model_pe)])
        st.line_chart(pe_df)

        st.info("""
        🧠 **Interpretation:**  
        - Lower dimensions vary slowly → capture global position patterns.  
        - Higher dimensions oscillate rapidly → encode fine-grained positional info.  
        - Each position has a unique sinusoidal signature.
        """)

    st.markdown("---")

    # Embedding dimension
    d_model = st.slider("Embedding dimension (d_model)", 2, 64, 5)
  

    # Create random embeddings for each token
    matrix = np.random.randn(len(tokens), d_model)

    # Convert to DataFrame for labeled display
    df = pd.DataFrame(matrix, index=tokens, columns=[f"Dim {i+1}" for i in range(d_model)])

    st.write("Embedding Matrix (Token + PE → Embedding):")
    st.dataframe(df, use_container_width=True)

    # Random weight matrices for Q, K, V (simulating learned parameters)
    W_q = np.random.randn(d_model, d_model)
    W_k = np.random.randn(d_model, d_model)
    W_v = np.random.randn(d_model, d_model)

    X = df.to_numpy()

    # Compute Q, K, V
    Q = X @ W_q
    K = X @ W_k
    V = X @ W_v

    # Convert to DataFrames for display
    df_Q = pd.DataFrame(Q, columns=[f"dim{i+1}" for i in range(d_model)])
    df_K = pd.DataFrame(K, columns=[f"dim{i+1}" for i in range(d_model)])
    df_V = pd.DataFrame(V, columns=[f"dim{i+1}" for i in range(d_model)])

    st.markdown("---")

    st.subheader("Query Matrix (Q)")
    st.dataframe(df_Q, use_container_width=True)

    st.subheader("Key Matrix (K)")
    st.dataframe(df_K, use_container_width=True)

    st.subheader("Value Matrix (V)")
    st.dataframe(df_V, use_container_width=True)

    if st.checkbox("Show Scaled Dot-Product Steps"):
        raw_scores = np.dot(Q, K.T)
        scaled_scores = raw_scores / np.sqrt(d_model)
        st.write("**Raw Scores (QKᵀ):**")
        st.dataframe(pd.DataFrame(raw_scores, index=unique_tokens, columns=unique_tokens))
        st.write("**Scaled Scores (QKᵀ / √d):**")
        st.dataframe(pd.DataFrame(scaled_scores, index=unique_tokens, columns=unique_tokens))


    attention_weight = softmax(np.dot(Q, K.T) / np.sqrt(d_model))
    attention_weight = pd.DataFrame(attention_weight, index = unique_tokens, columns = unique_tokens)

    

    st.markdown("---")

    st.subheader("Attention Weights ")
    st.dataframe(attention_weight, use_container_width=True)

    # -------------------------------
    # Explainable Mode (selectbox-based)
    # -------------------------------
    st.markdown("---")
    st.subheader("🧩 Explainable Mode (Token-Pair Narration)")

    # Choose a "from" token (who is paying attention)
    from_token = st.selectbox("Select source token (who attends)", ["None"] + unique_tokens)

    # If a from_token is chosen, let user pick a target; otherwise show info
    if from_token == "None":
        st.info("Select a source token to inspect which tokens it attends to. Or click 'Top target' to auto-select the highest-attended token.")
    else:
        # button to auto-select top target
        if st.button("🔎 Show Top target for selected source"):
            top_target = attention_weight.loc[from_token].idxmax()
            to_token = top_target
        else:
            to_token = st.selectbox("Select target token (to)", ["None"] + unique_tokens, index=0)

        # If user chose a target, display narration and highlight cell
        if to_token is not None and to_token != "None":
            i = attention_weight.index.get_loc(from_token)
            j = attention_weight.columns.get_loc(to_token)
            weight = float(attention_weight.iloc[i, j])

            st.success(f"🔎 **'{from_token}'** pays **{weight*100:.2f}%** attention to **'{to_token}'**.")
        else:
            st.info("Choose a target token to see the attention percentage and a highlighted heatmap cell.")


    with st.expander("📘 Scaled Dot-Product Attention Formula"):
        st.latex(r"Attention Weights(Q, K) = softmax\left(\frac{QK^T}{\sqrt{d_k}}\right)")

    st.markdown("---")

    st.subheader("🔥 Attention Heatmap")

    cmaps = st.selectbox(
        "Select Color Map for Heatmap",
        ["YlGnBu", "coolwarm", "viridis", "magma", "plasma", "rocket", "flare", "crest", "icefire", "Spectral"],
        index=0
    )
    annot = st.toggle("Show numeric weights on heatmap", value=False)

    # Token selector (optional)
    selected_token = st.selectbox("🎯 Highlight Token (optional)", ["None"] + unique_tokens)

    fig, ax = plt.subplots()
    # ax.set_facecolor("black" if theme == "Dark" else "white")

    n = attention_weight.shape[0]

    # --- CASE 1: No token selected -> Show full heatmap ---
    if selected_token == "None":
        sns.heatmap(attention_weight, annot=annot, cmap=cmaps, ax=ax)
        ax.set_xticklabels(attention_weight.columns, rotation=45, ha='right')
        ax.set_yticklabels(attention_weight.index, rotation=0)
        ax.set_xlim(0, n)
        ax.set_ylim(n, 0)

    # --- CASE 2: Highlight one token's entire row & column ---
    else:
        idx = attention_weight.index.get_loc(selected_token)

        # Build mask: True = hide, False = show
        rows = np.arange(n)[:, None]   # shape (n,1)
        cols = np.arange(n)[None, :]   # shape (1,n)
        mask = ~( (rows == idx) | (cols == idx) )  # show row idx and col idx

        sns.heatmap(attention_weight, annot=annot, cmap=cmaps, ax=ax, mask=mask)
        ax.set_xticklabels(attention_weight.columns, rotation=45, ha='right')
        ax.set_yticklabels(attention_weight.index, rotation=0)
        ax.set_xlim(0, n)
        ax.set_ylim(n, 0)

        # Optional: draw rectangle outlines around the selected row & column for clarity
        # Horizontal rectangle covering the selected row
        ax.add_patch(plt.Rectangle((0, idx), n, 1, fill=False, edgecolor='red', lw=1.5))
        # Vertical rectangle covering the selected column
        ax.add_patch(plt.Rectangle((idx, 0), 1, n, fill=False, edgecolor='red', lw=1.5))

    st.pyplot(fig)


    if selected_token != "None":
        st.markdown(f"### 📈 Attention distribution for **{selected_token}**")
        attn_row = attention_weight.loc[selected_token]
        fig2, ax2 = plt.subplots(edgecolor = "black", facecolor = "white")
        ax2.bar(attn_row.index, attn_row.values, color = "gray", edgecolor = "black")
        ax2.set_xlabel("Tokens attended to")
        ax2.set_ylabel("Attention Weight")
        ax2.set_title(f"Attention of '{selected_token}' → Other Tokens")
        plt.xticks(rotation=45)
        st.pyplot(fig2)

    
    st.markdown("---")
    st.subheader("🕸️ Interactive Token Relationship Graph")

    # Token selection
    focus_token = st.selectbox("🎯 Select a token to visualize its attention relations", unique_tokens)

    # Attention threshold
    threshold = st.slider("Attention weight threshold", 0.0, 1.0, 0.05, step=0.001)

    # Create a directed graph (using positions similar to networkx spring layout)
    import networkx as nx
    G = nx.DiGraph()

    # Add nodes
    for token in unique_tokens:
        G.add_node(token)

    # Get focus token index
    focus_idx = attention_weight.index.get_loc(focus_token)

    # Extract attention relations (row for selected token)
    relations = attention_weight.iloc[focus_idx]

    # Add edges with weights
    for target, weight in relations.items():
        if weight > threshold:
            G.add_edge(focus_token, target, weight=weight)

    # Use spring layout for node positioning
    pos = nx.spring_layout(G, seed=42, k=0.5)

    # Build Plotly edge traces
    edge_x, edge_y, edge_weights, edge_labels = [], [], [], []
    for u, v, data in G.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
        edge_weights.append(data['weight'])
        edge_labels.append(f"{u} → {v}: {data['weight']:.3f}")

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=2, color='gray'),
        hoverinfo='none',
        mode='lines'
    )

    # Build node traces
    node_x, node_y, node_text = [], [], []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)

    # node_trace = go.Scatter(
    #     x=node_x, y=node_y,
    #     mode='markers+text',
    #     hoverinfo='text',
    #     text=node_text,
    #     textposition="bottom center",
    #     marker=dict(
    #         showscale=True,
    #         colorscale='YlGnBu',
    #         color=[attention_weight.loc[focus_token, node] if node != focus_token else 1.0 for node in G.nodes()],
    #         size=[30 if node == focus_token else 18 for node in G.nodes()],
    #         colorbar=dict(
    #             title="Attention Weight",
    #             thickness=15,
    #             xanchor='left',
    #             titleside='right'
    #         ),
    #         line=dict(width=2, color='white')
    #     )
    # )

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=node_text,
        textposition="bottom center",
        marker=dict(
            showscale=True,                # enables colorbar
            colorscale='YlGnBu',           # valid Plotly colormap
            reversescale=True,
            # convert to native floats
            color=[float(attention_weight.loc[focus_token, node])
                   if node != focus_token else 1.0
                   for node in G.nodes()],
            size=[30 if node == focus_token else 18 for node in G.nodes()],
            # ✅ simplified and compliant colorbar (no nested dicts)
            colorbar=dict(
                title="Attention Weight",  # plain string title works on all versions
                thickness=15,
                len=0.5,
                x=1.05
            ),
            line=dict(width=2, color='white')
        )
    )




    # Combine everything
    fig3 = go.Figure(data=[edge_trace, node_trace],
        layout=go.Layout(
            title=f"Token Relationships for '{focus_token}' (Threshold > {threshold:.2f})",
            title_x=0.5,
            showlegend=False,
            hovermode='closest',
            margin=dict(b=0, l=0, r=0, t=40),
            xaxis=dict(showgrid=False, zeroline=False, visible=False),
            yaxis=dict(showgrid=False, zeroline=False, visible=False),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
        )
    )

    st.plotly_chart(fig3, use_container_width=True)

    st.markdown("---")

    st.title("**🔍 Attention Summary**")
    st.dataframe(pd.DataFrame({
        "Mean": attention_weight.mean(axis=1),
        "Max": attention_weight.max(axis=1),
        "Entropy": -np.sum(attention_weight * np.log(attention_weight + 1e-9), axis=1)
    }, index=unique_tokens))

    st.subheader("🎯 Token Focus Visualization (via Entropy)")

    entropy = -np.sum(attention_weight * np.log(attention_weight + 1e-9), axis=1)
    fig_ent, ax_ent = plt.subplots()
    ax_ent.bar(unique_tokens, entropy, color='gray', edgecolor='black')
    ax_ent.set_ylabel("Entropy")
    ax_ent.set_title("Lower = Focused Attention, Higher = Diffused Attention")
    plt.xticks(rotation=45)
    st.pyplot(fig_ent)


    st.markdown("---")

    with st.expander("📘 Final Attention Formula"):
        st.latex(r"Attention(Q, K, V) = softmax\left(\frac{QK^T}{\sqrt{d_k}}\right)V")

    output = attention_weight.to_numpy() @ V
    st.subheader("Final Attention Output (Attention × V)")
    st.dataframe(pd.DataFrame(output, index=unique_tokens))



    st.markdown("---")
    st.subheader("🔍 Cosine Similarity Between Tokens (Q · K)")

    similarity = np.dot(Q, K.T) / (
        np.linalg.norm(Q, axis=1, keepdims=True) * np.linalg.norm(K, axis=1, keepdims=True)
    )
    similarity_df = pd.DataFrame(similarity, index=unique_tokens, columns=unique_tokens)

    cmaps_sim = st.selectbox(
        "Select Colormap for Similarity Matrix",
        ["YlGnBu", "coolwarm", "viridis", "magma", "plasma", "rocket", "flare", "crest", "icefire", "Spectral"],
        index=0
    )
    fig_sim, ax_sim = plt.subplots()
    sns.heatmap(similarity_df, cmap=cmaps_sim, annot=False, ax=ax_sim)
    ax_sim.set_title("Cosine Similarity (Q vs K)")
    st.pyplot(fig_sim)






















