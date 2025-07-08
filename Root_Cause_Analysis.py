import json
from collections import Counter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

# Load logs.json
with open('logs.json', 'r') as f:
    logs = json.load(f)

# Prepare DataFrame
rows = []
for entry in logs:
    query_id = entry['query_id']
    latency = entry['response_latency_ms']
    user_feedback = entry.get('user_feedback', None)
    retrieved_chunks = entry.get('retrieved_chunks', [])
    num_chunks = len(retrieved_chunks)
    sources = [chunk['source'] for chunk in retrieved_chunks]
    source_counts = Counter(sources)
    avg_retrieval_score = np.mean([chunk['retrieval_score'] for chunk in retrieved_chunks]) if retrieved_chunks else np.nan
    
    rows.append({
        'query_id': query_id,
        'latency_ms': latency,
        'user_feedback': user_feedback,
        'num_retrieved_chunks': num_chunks,
        'avg_retrieval_score': avg_retrieval_score,
        'sources': sources,
        'pdf_chunks': source_counts.get('Archived Design Docs (PDFs)', 0),
        'wiki_chunks': source_counts.get('Engineering Wiki', 0),
        'conf_chunks': source_counts.get('Confluence', 0),
        'pdf_avg_score': np.mean([chunk['retrieval_score'] for chunk in retrieved_chunks if chunk['source'] == 'Archived Design Docs (PDFs)']) if source_counts.get('Archived Design Docs (PDFs)', 0) > 0 else np.nan,
        'wiki_avg_score': np.mean([chunk['retrieval_score'] for chunk in retrieved_chunks if chunk['source'] == 'Engineering Wiki']) if source_counts.get('Engineering Wiki', 0) > 0 else np.nan,
        'conf_avg_score': np.mean([chunk['retrieval_score'] for chunk in retrieved_chunks if chunk['source'] == 'Confluence']) if source_counts.get('Confluence', 0) > 0 else np.nan,
    })

df = pd.DataFrame(rows)

# Define mixed_sources: True if more than one unique source in sources list
df['mixed_sources'] = df['sources'].apply(lambda s: len(set(s)) > 1)

# Basic stats
print("Latency stats (ms):")
print(df['latency_ms'].describe())

# Slow queries
slow_queries = df[df['latency_ms'] > 3500]
print(f"Number of slow queries (>3500ms): {len(slow_queries)}")

# Source counts in slow queries
slow_sources = []
for sources in slow_queries['sources']:
    slow_sources.extend(sources)
slow_source_counts = Counter(slow_sources)
print("Source counts for slow queries:")
print(slow_source_counts)

# Negative feedback queries
neg_feedback = df[df['user_feedback'] == 'thumb_down']
print(f"Number of negative feedback queries: {len(neg_feedback)}")

neg_sources = []
for sources in neg_feedback['sources']:
    neg_sources.extend(sources)
neg_source_counts = Counter(neg_sources)
print("Source counts for negative feedback queries:")
print(neg_source_counts)

# Overall source counts
all_sources = []
for sources in df['sources']:
    all_sources.extend(sources)
all_source_counts = Counter(all_sources)
print("\nOverall source counts:")
print(all_source_counts)

# Latency with/without PDFs
df['has_pdf_chunk'] = df['pdf_chunks'] > 0
print(f"\nAverage latency with PDF chunks: {df[df['has_pdf_chunk']]['latency_ms'].mean():.2f} ms")
print(f"Average latency without PDF chunks: {df[~df['has_pdf_chunk']]['latency_ms'].mean():.2f} ms")

# Negative feedback rate with/without PDFs
print(f"Negative feedback rate with PDF chunks: {df[df['has_pdf_chunk']]['user_feedback'].eq('thumb_down').mean():.2%}")
print(f"Negative feedback rate without PDF chunks: {df[~df['has_pdf_chunk']]['user_feedback'].eq('thumb_down').mean():.2%}")

# --- Additional Analyses ---

# 1. Latency vs number of PDF + Confluence chunks
df['pdf_conf_chunks'] = df['pdf_chunks'] + df['conf_chunks']
plt.scatter(df['pdf_conf_chunks'], df['latency_ms'])
plt.xlabel('Number of PDF + Confluence Chunks')
plt.ylabel('Response Latency (ms)')
plt.title('Latency vs PDF + Confluence Chunk Count')
plt.show()

# 2. Retrieval scores by source in negative vs positive feedback queries
def print_retrieval_score_comparison(source):
    pos_scores = df[df['user_feedback'] == 'thumb_up'][f'{source}_avg_score'].dropna()
    neg_scores = df[df['user_feedback'] == 'thumb_down'][f'{source}_avg_score'].dropna()
    print(f"\nRetrieval scores for {source} - Positive feedback mean: {pos_scores.mean():.3f}, Negative feedback mean: {neg_scores.mean():.3f}")
    if len(pos_scores) > 1 and len(neg_scores) > 1:
        t_stat, p_val = ttest_ind(pos_scores, neg_scores, equal_var=False)
        print(f"T-test p-value: {p_val:.4f} (significant if <0.05)")

print_retrieval_score_comparison('pdf')
print_retrieval_score_comparison('conf')
print_retrieval_score_comparison('wiki')

# 3. Distribution of retrieval scores per source overall
plt.boxplot([
    df['pdf_avg_score'].dropna(),
    df['conf_avg_score'].dropna(),
    df['wiki_avg_score'].dropna()
], tick_labels=['PDF', 'Confluence', 'Wiki'])
plt.title('Distribution of Retrieval Scores by Source')
plt.ylabel('Retrieval Score')
plt.show()

# 4. Negative feedback rate for mixed vs single-source queries
mixed_neg_rate = df[df['mixed_sources']]['user_feedback'].eq('thumb_down').mean()
single_source_queries = df[~df['mixed_sources']]
single_neg_rate = single_source_queries['user_feedback'].eq('thumb_down').mean()

print(f"Negative feedback rate with mixed sources: {mixed_neg_rate:.2%}")
print(f"Number of single-source queries: {len(single_source_queries)}")
print(f"Negative feedback rate with single source: {single_neg_rate:.2%}")

# Debugging NaN negative feedback rate for single-source queries
single_source_with_feedback = single_source_queries['user_feedback'].dropna()
print(f"Single-source queries with feedback: {len(single_source_with_feedback)}")
if len(single_source_with_feedback) > 0:
    neg_rate_single = single_source_with_feedback.eq('thumb_down').mean()
    print(f"Negative feedback rate for single-source queries (with feedback only): {neg_rate_single:.2%}")
else:
    print("No single-source queries with feedback found.")

# 5. Updated correlation matrix including chunk counts by source
corr = df[['latency_ms', 'num_retrieved_chunks', 'pdf_chunks', 'conf_chunks', 'wiki_chunks', 'avg_retrieval_score']].corr()
print("\nUpdated correlation matrix:")
print(corr)

import seaborn as sns

# Boxplot of retrieval scores by source
plt.figure(figsize=(8,6))
sns.boxplot(data=df.melt(id_vars=['query_id'], 
                         value_vars=['pdf_avg_score', 'conf_avg_score', 'wiki_avg_score']),
            x='variable', y='value')
plt.xticks(ticks=[0,1,2], labels=['PDF', 'Confluence', 'Wiki'])
plt.ylabel('Average Retrieval Score')
plt.title('Distribution of Retrieval Scores by Source')
plt.show()

# Compare retrieval scores in positive vs negative feedback queries
for source in ['pdf_avg_score', 'conf_avg_score', 'wiki_avg_score']:
    pos_scores = df[df['user_feedback'] == 'thumb_up'][source].dropna()
    neg_scores = df[df['user_feedback'] == 'thumb_down'][source].dropna()
    print(f"{source} - Pos mean: {pos_scores.mean():.3f}, Neg mean: {neg_scores.mean():.3f}")
    if len(pos_scores) > 1 and len(neg_scores) > 1:
        from scipy.stats import ttest_ind
        t_stat, p_val = ttest_ind(pos_scores, neg_scores, equal_var=False)
        print(f"T-test p-value: {p_val:.4f} (significant if <0.05)")

# Correlation between retrieval scores and latency
for source in ['pdf_avg_score', 'conf_avg_score', 'wiki_avg_score']:
    corr = df[[source, 'latency_ms']].dropna().corr().iloc[0,1]
    print(f"Correlation between {source} and latency: {corr:.3f}")
