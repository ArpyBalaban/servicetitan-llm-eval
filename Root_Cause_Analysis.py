import json
from collections import Counter, defaultdict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load logs.json
with open('logs.json', 'r') as f:
    logs = json.load(f)

# Convert to a DataFrame for easier analysis
rows = []
for entry in logs:
    query_id = entry['query_id']
    latency = entry['response_latency_ms']
    user_feedback = entry.get('user_feedback', None)
    retrieved_chunks = entry.get('retrieved_chunks', [])
    num_chunks = len(retrieved_chunks)
    sources = [chunk['source'] for chunk in retrieved_chunks]
    avg_retrieval_score = np.mean([chunk['retrieval_score'] for chunk in retrieved_chunks]) if retrieved_chunks else np.nan
    rows.append({
        'query_id': query_id,
        'latency_ms': latency,
        'user_feedback': user_feedback,
        'num_retrieved_chunks': num_chunks,
        'avg_retrieval_score': avg_retrieval_score,
        'sources': sources
    })

df = pd.DataFrame(rows)

# Latency stats
print("Latency stats (ms):")
print(df['latency_ms'].describe())

# Define slow queries as those exceeding 3500 ms SLA (3.5 sec)
slow_queries = df[df['latency_ms'] > 3500]
print(f"Number of slow queries (>3500ms): {len(slow_queries)}")

# Analyze sources for slow queries
slow_sources = []
for sources in slow_queries['sources']:
    slow_sources.extend(sources)
slow_source_counts = Counter(slow_sources)
print("Source counts for slow queries:")
print(slow_source_counts)

# Analyze negative feedback queries (likely incorrect/outdated answers)
neg_feedback = df[df['user_feedback'] == 'thumb_down']
print(f"Number of negative feedback queries: {len(neg_feedback)}")

neg_sources = []
for sources in neg_feedback['sources']:
    neg_sources.extend(sources)
neg_source_counts = Counter(neg_sources)
print("Source counts for negative feedback queries:")
print(neg_source_counts)

# Correlations
print("\nCorrelation matrix:")
corr_matrix = df[['latency_ms', 'num_retrieved_chunks', 'avg_retrieval_score']].corr()
print(corr_matrix)

# Plot latency histogram
plt.hist(df['latency_ms'], bins=30)
plt.xlabel('Response latency (ms)')
plt.ylabel('Number of queries')
plt.title('Response latency distribution')
plt.show()

# Optional: Plot source distribution in all queries
all_sources = []
for sources in df['sources']:
    all_sources.extend(sources)
all_source_counts = Counter(all_sources)
print("\nOverall source counts:")
print(all_source_counts)

# Additional: Check if queries with PDFs tend to be slower or have negative feedback
df['has_pdf_chunk'] = df['sources'].apply(lambda s: 'Archived Design Docs (PDFs)' in s)
pdf_queries = df[df['has_pdf_chunk']]
non_pdf_queries = df[~df['has_pdf_chunk']]

print(f"\nAverage latency with PDF chunks: {pdf_queries['latency_ms'].mean():.2f} ms")
print(f"Average latency without PDF chunks: {non_pdf_queries['latency_ms'].mean():.2f} ms")
print(f"Negative feedback rate with PDF chunks: {pdf_queries['user_feedback'].eq('thumb_down').mean():.2%}")
print(f"Negative feedback rate without PDF chunks: {non_pdf_queries['user_feedback'].eq('thumb_down').mean():.2%}")
