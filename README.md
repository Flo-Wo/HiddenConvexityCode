# Hidden Convexity

This repository provides the official code for the paper:  

**"Global Solutions to Non-Convex Functional Constrained Problems with Hidden Convexity"**  

by Fatkhullin I., Lan G., He N., Wolf F., 2025

We implement the optimization algorithms and problem formulations introduced in the paper, 
and provide scripts to reproduce all experimental figures.

---

## Getting Started

### Requirements
- Python 3.10+, e.g. Python 3.10.14
- numpy, matplotlib, scipy (see `requirements.txt` for the complete list)

### Installation
```bash
git clone git@github.com:Flo-Wo/HiddenConvexityCode.git
cd HiddenConvexityCode
pip install -r requirements.txt
```

## Reproducing Experiments
All experiments can be reproduced directly from the /src directory using the makefile.
For **example**, to recreate Figure 1(a,b):
```shell
cd src
make figure_1_ab
```

## Repository Structure
```
/src
├── makefile            # Commands to recreate all paper figures
├── algorithms.py       # Optimization algorithms introduced in the paper
├── base.py             # Abstract problem + optimization algorithm result formulation
├── colors_setting.py   # Colors and labels for plots
├── create_plots.py     # Utility functions for figures in 𝓧 and 𝓤 space
├── parser.py           # Command line argument parser
├── <>_example.py       # Problem formulation (matches base.py)
└── main.py             # Entry point to run experiments
└── figs/               # All figures used in the paper
```

## Citation
If you use this code, please cite our work:
```bibtex
@misc{fatkhullin2025globalsolutionsnonconvexfunctional,
      title={Global Solutions to Non-Convex Functional Constrained Problems with Hidden Convexity}, 
      author={Ilyas Fatkhullin and Niao He and Guanghui Lan and Florian Wolf},
      year={2025},
      eprint={2511.10626},
      archivePrefix={arXiv},
      primaryClass={math.OC},
      url={https://arxiv.org/abs/2511.10626}, 
}
```

## Contact
Maintained by F. Wolf. For questions or issues, please open a GitHub issue.

