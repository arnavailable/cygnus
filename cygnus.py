# Download the required packages
# pip install -r requirements.txt

##############################################################

# Import all necessary libraries
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import networkx as nx
import snowflake.connector
from pyvis.network import Network

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
    SELECT 
        sender_account AS start_account,
        beneficiary_account AS current_account,
        transaction_amount AS start_amount,
        transaction_amount AS current_amount,
        transaction_date AS start_date,
        transaction_date AS current_date,
        ARRAY_CONSTRUCT(sender_account, beneficiary_account) AS account_path,
        ARRAY_CONSTRUCT(transaction_date) AS date_path,
        ARRAY_CONSTRUCT(transaction_amount) AS amount_path,
        ARRAY_CONSTRUCT(transaction_id) AS id_path,
        1 AS depth
    FROM "CYGNUS"."PUBLIC"."TRANSACTIONS"

    UNION ALL

    SELECT 
        tp.start_account,
        t.beneficiary_account,
        tp.start_amount,
        t.transaction_amount,
        tp.start_date,
        t.transaction_date,
        ARRAY_APPEND(tp.account_path, t.beneficiary_account),
        ARRAY_APPEND(tp.date_path, t.transaction_date),
        ARRAY_APPEND(tp.amount_path, t.transaction_amount),
        ARRAY_APPEND(tp.id_path, t.transaction_id),
        tp.depth + 1
    FROM transaction_paths tp,
         LATERAL (
            SELECT *
            FROM "CYGNUS"."PUBLIC"."TRANSACTIONS" t
            WHERE tp.current_account = t.sender_account
            AND t.transaction_date > tp.current_date
            AND t.transaction_amount BETWEEN tp.current_amount * 0.9 AND tp.current_amount * 1.1
            AND NOT ARRAY_CONTAINS(tp.account_path, ARRAY_CONSTRUCT(t.beneficiary_account))
        ) t
)
SELECT account_path, date_path, amount_path, id_path, depth
FROM transaction_paths
WHERE current_account = start_account
AND depth > 2
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
df.columns = ['path', 'date_path', 'amount_path', 'id_path', 'depth']

# Clean path, date_path, and amount_path columns by removing \n from array
df['path'] = df['path'].apply(lambda x: eval(x))
df['date_path'] = df['date_path'].apply(lambda x: eval(x))
df['amount_path'] = df['amount_path'].apply(lambda x: eval(x))
df['id_path'] = df['id_path'].apply(lambda x: eval(x))

# Expand lists into rows while keeping the original index
rows = []
for idx, row in df.iterrows():
    for i in range(len(row["date_path"])):
        rows.append([
            idx,  # Preserve original index
            row["id_path"][i],
            row["path"][i],
            row["path"][i + 1],
            row["date_path"][i],
            row["amount_path"][i]
        ])

# Create new DataFrame
df = pd.DataFrame(rows, columns=["index", "transaction_id", "sender_account", "beneficiary_account", "transaction_date", "transaction_amount"]).set_index("index")

# Calculate %diff from previous row if index is same
df['percentage_difference'] = round(df.groupby('index')['transaction_amount'].pct_change()*100, 4)
df['percentage_difference'] = df['percentage_difference'].fillna(0)

##############################################################

# Set header title
st.title('Visualization of Transaction Loops')

# Define list of selection options
suspicious_accounts = df.sender_account.unique()
suspicious_accounts.sort()

# Implement multiselect dropdown menu for option selection (returns a list)
selected_account = st.selectbox('Select account number', suspicious_accounts)

# Set info message on initial site load
if not selected_account:
    st.text('Choose an account number to start')

# Create network graph when user selects >= 1 item
else:
    df_select = df[df.index == df[df.sender_account == selected_account].index[0]]

    # Create a directed graph
    G = nx.DiGraph()

    # Add nodes and edges with attributes
    for _, row in df_select.iterrows():
        G.add_node(row["sender_account"], label=str(row["sender_account"]), color="#4CAF50")  # Green sender
        G.add_node(row["beneficiary_account"], label=str(row["beneficiary_account"]), color="#2196F3")  # Blue receiver
        G.add_edge(
            row["sender_account"], row["beneficiary_account"],
            title=f"ID: {row['transaction_id']}\nAmount: ${row['transaction_amount']}\nDate: {row['transaction_date']}\nChange: {row['percentage_difference']}%",  # Tooltip when hovering
            label=f"{row['transaction_id']}\n${row['transaction_amount']}\n{row['transaction_date']}\n{row['percentage_difference']}",  # Visible label
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

    # Legend using Markdown with HTML
    st.markdown(
        '<div style="display: flex; align-items: center; gap: 15px;">' +
        ''.join(f'<div style="display: flex; align-items: center; gap: 5px;">'
                f'<div style="width: 15px; height: 15px; background-color: {color}; border-radius: 50%;"></div>'
                f'<span>{label}</span></div>'
                for label, color in [("Origination Account", "#2196F3"),
                                    ("Intermediary Accounts", "#4CAF50"),
                                    ("Transaction Edge", "#FF9800")]) +
        '</div>', unsafe_allow_html=True)

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

    # Show the DataFrame
    st.write(df_select[['transaction_id', 'transaction_date', 'sender_account', 'beneficiary_account', 'transaction_amount', 'percentage_difference']].reset_index(drop=True))

##############################################################