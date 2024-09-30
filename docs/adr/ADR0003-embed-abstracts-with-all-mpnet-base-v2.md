# 3. Embed abstracts with all-mpnet-base-v2

## Status

| Status   | Time                         |
| -------- | ---------------------------- |
| Accepted | 2024-09-30T18:08:21.1968594Z |

## Context

To perform RAG operations, the documents need to be embedded first.

In order to do this, we can leverage existing PTMs focussed on embedding natural
language documents.

## Decision

Leveraging `sentence-transformers` via `pytorch` with the Intel Extensions
enabled, we will utilize the `all-mpnet-base-v2` model to embed the *abstracts*
of arXiv documents.

These embeddings will be stored in a vector database.

## Consequences

We will need to create a persistent vector database to hold all of the
embeddings.

Additionally, our embeddings will be limited to those that `all-mpnet-base-v2`
generate. This includes both our user query and arXiv document abstracts.
