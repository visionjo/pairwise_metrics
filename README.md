# Pairwise Metrics and Visualization
![ql-1]






## Table of Contents
* [Pairwise Metrics and Visualization](#Pairwise Metrics and Visualization)

[**Overview**](#overview)

[**Motivation**](#motivation)

[**Use Cases**](#motivation)
  * [**1 - Cluster Evaluation**](#case-1-cluster-evaluation)
  * [**2 - Visualize Pairwise Relationships**](#case-2-pairwise-relationships)
  
[**Features**](#features)
  * [**Metrics**](#metrics)
  * [**Visualization**](#patched-fonts)
  * [**Tests**](#combinations)

[**Project Motivation**](#project-motivation)

**Additional Info**
  * [**Changelog**](#changelog)
  * [**License**](#license)

## Overview
Evaluation and visualization tools for pairwise measures. The motivation for this repo is to evaluate clustering 
algorithms. However, amongst other use cases, this code-base is relevant in problems that involve sample pairs and 
distance matrices.

## Motivation
SKLearn is complete with many metrics, which include submodules for 
[clustering](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics.cluster) and
[pairwise](https://scikit-learn.org/stable/modules/classes.html#pairwise-metrics). However, this code-base sets out to 
compliment this with pairwise metrics determined for cluster (or class) assignments of arbitrary assignments.
 
## Use Cases

### `Case 1: Cluster Evaluation`

> Measure performance for cluster assignments (i.e., pseudo-labels) provided ground-truth labels.

Various pair-wise metrics allow for in depth analysis of clustering algorithms.


### `Case 2: Visualize Pairwise Relationships`

> Powerful visualizations generated on-the-fly provide quick and systematic way to analyze and communicate results.

Visualizations tools and demos included.

## Screenshots
TODO

## Tech/framework used
Ex. -

<b>Built with</b>
- [Electron](https://electron.atom.io)

## Features
TODO

## Code Example
TODO

## Installation
TODO

## Tests
TODO

## How to use?
TODO

## Contribute
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## Related Links
- [Pairwise clustering tools (MATLAB and C++)][agglomerative] 
- [Clustering Algorithms and Metrics][stanford-resource]
[ql-1]
## Action Items
This to do. Unless specified, order is arbitrary.
- [x] Update LICENSE
- [ ] Complete README
- [x] Utility functions
- [ ] Visualization Tools
- [ ] Make metrics a class
    - [ ] inherent visualization tools 
- [ ] Demos and notebooks
- [ ] Tests
    - [x] Choose interface (i.e., python package) and create skeleton code-base
    - [x] Unit tests
    - [ ] Type checking
        - [ ] Inputs
        - [ ] Outputs
    - [ ] All tests
    - [ ] Add script for auto-testing
        - [ ] Decide on interface/ technology for this
        - [ ] Create material providing/ setting relevant configurations
- [ ] Add pre-commit for git (i.e., run tests before commit, and only do if tests PASS)
- [ ] Contributing Guidelines (create)

## Resources

## Changelog

See [changelog.md](changelog.md)

## License

[MIT](LICENSE) © [Joseph Robinson](https://www.jrobsvision.com/) 


<!--
Repo References
-->

[agglomerative]:https://github.com/visionjo/Agglomerative_Clustering.git
[repo]: https://github.com/visionjo/pairwise_metrics

<!--
Website References
-->

[stanford-resource]:https://www.ims.uni-stuttgart.de/institut/mitarbeiter/schulte/theses/phd/algorithm.pdf

<!--
Link References
-->
[release]:https://github.com/visionjo/pairwise_metrics/releases/latest "Latest Release (external link) ➶"

<!--
Quick Link Images
-->
[ql-1]:images/clustering_pairs_graphic.png "Toy view of a clustering ➶"