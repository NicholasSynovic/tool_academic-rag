# 2. Use arXiv dataset

## Status

| Status   | Time                         |
| -------- | ---------------------------- |
| Accepted | 2024-09-30T17:23:31.5921765Z |

## Context

This application will perform Retrieval Augmented Generation tasks on a
pre-generated dataset.

## Decision

The dataset to leverage for this application is the arXiv dataset availible on
Kaggle [here](https://www.kaggle.com/datasets/Cornell-University/arxiv).

I have already written code to convert the raw dataset to an SQLite database
[here](https://github.com/NicholasSynovic/tool_arXiv-db).

## Consequences

Require that my project is a dependency that `poetry` manages going forward.

Support reading from the database, ideally with Intel
[oneAPI AI Toolkit tech](https://www.intel.com/content/www/us/en/developer/tools/oneapi/ai-analytics-toolkit.html).
