# Snippet Finder

This project implements an algorithm for discovering representative patterns (snippets) in time series data using parallel computations on a GPU.

## Introduction

The Snippet Finder is designed to identify representative patterns in time series data. This algorithm leverages parallel computations to enhance efficiency, making it suitable for various applications such as biometric data analysis, physical activity monitoring, and more.

## Dependencies

The project requires the following libraries:

- `cupy`: For numerical computations on the GPU.
- `numpy`: For numerical computations.
- `matplotlib`: For data visualization.
- `dask`: For parallel computing.
- `stumpy`: For time series analysis.
- `pandas`: For data manipulation.
- `scikit-learn` and `scikit-learn-extra`: For machine learning algorithms.

## Installation

1. Clone the repository:

    ```sh
    git clone https://github.com/your-username/your-repository.git
    cd your-repository
    ```

2. Create and activate a virtual environment:

    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. Install the dependencies:

    ```sh
    pip install cupy numpy matplotlib dask stumpy pandas scikit-learn scikit-learn-extra
    ```

## Usage

Open the `parallelizedSnippetFinder.ipynb` file in a Jupyter Notebook environment and execute the cells to run the algorithm.

### Example

1. Load the time series data.
2. Execute the algorithm to find snippets.
3. Visualize the results using the plotting functions.

```python
# Example usage of the notebook functions

# Load data
df = pd.read_csv('your_file.csv')

# Execute PSF algorithm
subseq_size, snippets, snippets_profiles, fractions, areas, profiles = find_snippets_with_psf(df, "your_column")

# Plot results
plot_snippets(df["time"], df["your_column"].values, snippets, subseq_size, fractions)
Theoretical Background
```

## About This Project

### Algorithm Overview
The Snippet Finder algorithm aims to find representative subsequences (snippets) within time series data. The core idea is to identify patterns that frequently occur and are significant across the dataset. This is particularly useful in applications where understanding the recurring patterns can provide insights into the underlying processes.

### Optimization Strategy
To optimize the computation of snippets, the MiniBatchKMeans strategy was implemented, which selects better distance profiles. This approach has reduced the execution time by almost 800x, with negligible loss in quality compared to the standard algorithm.

### Details of the Strategy
The use of MiniBatchKMeans allows for clustering in smaller batches of data, which is significantly faster and more memory-efficient than traditional KMeans. This method is particularly effective when dealing with large time series datasets, such as those used in this project.

### Advantages
Efficiency: The reduction in execution time is nearly 800 times compared to the traditional approach.
Quality: The quality of the identified snippets remains almost the same as the standard algorithm, ensuring that representative patterns are still identified with high precision.
Implementation
The implementation of this strategy can be seen in the Jupyter Notebook, where MiniBatchKMeans is used to optimize the snippet selection process.

```python
from sklearn.cluster import MiniBatchKMeans

# Example usage of MiniBatchKMeans
kmeans = MiniBatchKMeans(n_clusters=10, batch_size=1000)
kmeans.fit(data)
labels = kmeans.labels_
Parallel Computation
The algorithm uses cupy for parallel computations on a GPU, which significantly speeds up the process of calculating distance profiles and finding snippets. This is particularly beneficial for large datasets where computational efficiency is crucial.
```

## Contribution
Contributions are welcome! Feel free to open issues or submit pull requests.