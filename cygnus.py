# Download the required packages
# pip install -r requirements.txt

##############################################################

# Import all the necessary libraries
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import networkx as nx
import snowflake.connector
from pyvis.network import Network
import matplotlib.pyplot as plt

##############################################################

# Snowflake Connection
conn = snowflake.connector.connect(
    user = st.secrets["USER"],
    password = st.secrets["PASSWORD"],
    account = st.secrets["ACCOUNT"],
    warehouse = st.secrets["WAREHOUSE"],
    database = st.secrets["DATABASE"],
    schema = st.secrets["SCHEMA"]
)

# Query to get the data
query = """
WITH transaction_paths AS (
    -- Step 1: Start with all transactions as starting points
    SELECT 
        sender AS start_account,
        receiver AS current_account,
        amount AS start_amount,
        amount AS current_amount,
        date AS start_date,
        date AS current_date,
        ARRAY_CONSTRUCT(sender, receiver) AS path,
        ARRAY_CONSTRUCT(date) AS date_path,
        ARRAY_CONSTRUCT(amount) AS amount_path,
        1 AS depth
    FROM "CYGNUS"."PUBLIC"."LOOPS"

    UNION ALL

    -- Step 2: Extend paths by finding valid next transactions
    SELECT 
        tp.start_account,
        t.receiver,
        tp.start_amount,
        t.amount,
        tp.start_date,
        t.date,
        ARRAY_APPEND(tp.path, t.receiver),
        ARRAY_APPEND(tp.date_path, t.date),
        ARRAY_APPEND(tp.amount_path, t.amount),
        tp.depth + 1
    FROM transaction_paths tp,
         LATERAL (
            SELECT *
            FROM "CYGNUS"."PUBLIC"."LOOPS" t
            WHERE tp.current_account = t.sender
            AND t.date > tp.current_date
            AND t.amount BETWEEN tp.current_amount * 0.9 AND tp.current_amount * 1.1
            AND NOT ARRAY_CONTAINS(tp.path, ARRAY_CONSTRUCT(t.receiver)) -- Prevent cycles before completing loop
        ) t
)
SELECT path, date_path, amount_path, depth
FROM transaction_paths
WHERE current_account = start_account  -- Ensure loop completion
AND depth > 2  -- At least one intermediate transaction
ORDER BY start_date
"""

# Execute the query
cur = conn.cursor()
cur.execute(query)

# Fetch the data
rows = cur.fetchall()

# Create a dataframe
df = pd.DataFrame(rows, columns=[x[0] for x in cur.description])

# Close the cursor and connection
cur.close()
conn.close()

##############################################################

# Process the output
df.columns = ['path', 'date_path', 'amount_path', 'depth']

# Clean path, date_path, and amount_path columns by removing \n from array
df['path'] = df['path'].apply(lambda x: eval(x))
df['date_path'] = df['date_path'].apply(lambda x: eval(x))
df['amount_path'] = df['amount_path'].apply(lambda x: eval(x))

# Expand lists into rows while keeping the original index
rows = []
for idx, row in df.iterrows():
    for i in range(len(row["date_path"])):
        rows.append([
            idx,  # Preserve original index
            row["path"][i],
            row["path"][i + 1],
            row["date_path"][i],
            row["amount_path"][i]
        ])

# Create new DataFrame
df = pd.DataFrame(rows, columns=["index", "account_send", "account_receive", "date", "amount"]).set_index("index")

##############################################################

# Set header title
st.title('Network Graph Visualization of Transaction Loops')

# Define list of selection options
suspicious_accounts = df.account_send.unique()
suspicious_accounts.sort()

# Implement multiselect dropdown menu for option selection (returns a list)
selected_account = st.selectbox('Select account to visualize', suspicious_accounts)

# Set info message on initial site load
if not selected_account:
    st.text('Choose an account to start')

# Create network graph when user selects >= 1 item
else:
    df_select = df[df.index == df[df.account_send == selected_account].index[0]]

    # Create a directed graph
    G = nx.DiGraph()

    # Add nodes and edges with attributes
    for _, row in df_select.iterrows():
        G.add_node(row["account_send"], label=str(row["account_send"]), color="#4CAF50")  # Green sender
        G.add_node(row["account_receive"], label=str(row["account_receive"]), color="#2196F3")  # Blue receiver
        G.add_edge(
            row["account_send"], row["account_receive"],
            title=f"Amount: ${row['amount']}\nDate: {row['date']}",  # Tooltip when hovering
            label=f"${row['amount']}\n{row['date']}",  # Visible label
            color="#FF9800"  # Orange edges
        )

    # Create PyVis Network
    account_network = Network(height="500px", width="100%", directed=True, bgcolor="#222222", font_color="white")

    # Convert NetworkX graph to PyVis
    account_network.from_nx(G)

    # Improve layout with physics settings
    account_network.repulsion(node_distance=300, central_gravity=0.3, spring_length=100, spring_strength=0.1, damping=0.9)

    # Force edge labels to show
    for edge in account_network.edges:
        edge["font"] = {"size": 12, "color": "white", "background": "black"}

    # Save and read graph as HTML file (on Streamlit Sharing)
    try:
        path = '/tmp'
        account_network.save_graph(f'{path}/pyvis_graph.html')
        HtmlFile = open(f'{path}/pyvis_graph.html', 'r', encoding='utf-8')

    # Save and read graph as HTML file (locally)
    except:
        path = '/html_files'
        account_network.save_graph(f'{path}/pyvis_graph.html')
        HtmlFile = open(f'{path}/pyvis_graph.html', 'r', encoding='utf-8')

    # Load HTML file in HTML component for display on Streamlit page
    components.html(HtmlFile.read(), height=435)