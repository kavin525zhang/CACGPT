# Plot Generation

The plot module handles the plot generation from the data generated by the evaluation module.

You can use the `multirag-cli plot` command to generate plots to visualize the evaluation data.
The command comes with the following command line interface:
```
usage: multirag-cli plot [-h] [-d [DATA_PATH]] [-f [{pdf,png,svg}]] [-o [OUTPUT]]

Plotting

optional arguments:
  -h, --help            show this help message and exit
  -d [DATA_PATH], --data-path [DATA_PATH]
                        Path to the evaluation data file. (default: test-results.json)
  -f [{pdf,png,svg}], --format [{pdf,png,svg}]
                        Format for the plot files. (default: pdf)
  -o [OUTPUT], --output [OUTPUT]
                        Path to the output directory. (default: plots)
```


## Input Format
The data format for the json files with results is as follows:
```
{
    "standard-rag": {
        "1": {
            "success": [[...25 queries (int success)...], [...25 queries...], ...1-32 doc fetches],
            "success_ratio": [[...25 queries (float ratio)...], [...25 queries...], ...1-32 doc fetches],
            "category_success": [[...25 queries (int success)...], [...25 queries...], ...1-32 doc fetches]
            "category_success_ratio": [[...25 queries (float ratio)...], [...25 queries...], ...1-32 doc fetches]
        },
        "2": {...}
    },
    "multirag": {...},
    "split-rag": {...},
    "fusion-rag": {...},
    "fusion-multirag": {...},
    "multirag-strategy-decay": {...},
    "multirag-strategy-distance": {...},
    "split-rag-strategy-weighted": {...},
```