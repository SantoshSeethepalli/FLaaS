## Reference Documentation

This project implements core machine learning components inspired by foundational research papers. Each module closely follows the original algorithms and notation where applicable.

| Component                | Reference                         | Alignment with Paper                                                                                                                            |
| ------------------------ | --------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| `mlp.py`                 | Rumelhart et al. (1986), *Nature* | Implements a single hidden-layer MLP with sigmoid activation and backpropagation. Variable names and update rules follow the original notation. |
| `logistic_regression.py` | Cox (1958), *JRSS Series B*       | Implements binary logistic regression with sigmoid activation and log-likelihood gradient descent, as described in the paper.                   |
| `training.py`            | Bottou (2010), *COMPSTAT*         | Uses mini-batch stochastic gradient descent (SGD) and evaluation metrics per Bottou’s recommendations.                                          |
| `data_loader.py`         | Kohavi (1995), *IJCAI*            | Splits data into training/testing sets via random partitioning, as advised for accurate model evaluation.                                       |
| `base.py`, `__init__.py` | Pedregosa et al. (2011), *JMLR*   | Follows the scikit-learn-style API with an abstract base class and factory pattern for model instantiation.                                     |

### References

* Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). *Learning representations by back-propagating errors*. Nature, 323(6088), 533–536. [https://doi.org/10.1038/323533a0](https://doi.org/10.1038/323533a0)
* Cox, D. R. (1958). *The regression analysis of binary sequences*. Journal of the Royal Statistical Society: Series B (Methodological), 20(2), 215–242.
* Bottou, L. (2010). *Large-Scale Machine Learning with Stochastic Gradient Descent*. In COMPSTAT 2010 (pp. 177–186). [https://doi.org/10.1007/978-3-7908-2604-3\_16](https://doi.org/10.1007/978-3-7908-2604-3_16)
* Kohavi, R. (1995). *A study of cross-validation and bootstrap for accuracy estimation and model selection*. In IJCAI.
* Pedregosa, F., et al. (2011). *Scikit-learn: Machine Learning in Python*. Journal of Machine Learning Research, 12, 2825–2830.
