# Project description

Quantitative research of various markets.

This is a public demo version of the primary market research project (whose repo is private).

The description in Russian is available [here](https://github.com/AndreyRysin/market-research-demo/blob/main/docs/description_ru.md) (in `docs`).

## Intro

The project is a quantitative research in the field of time series analysis. The ultimate goal of the research is to develop an algorithm that predicts market price movements.

The research approach is based on the assumption that the market has predictable periods, and all the necessary information is already contained in the quotes. It is obvious that the market is sometimes deterministic: evident reactions to news, movements without corrections lasting longer than usual, and sudden “changes in behavior” that cannot be interpreted as random walk. Such anomalies may have preconditions that are, in fact, its initial phase.

The research is designed to confirm or reject the hypothesis about the existence of periods of predictable markets. And - if confirmed - learn to make a forecast in such periods. The market is characterized by a low signal-to-noise ratio, and a bit predictable periods (with a weak signal) alternate with unpredictable periods (without a signal). The first - less often, the second - more often. The main difficulty is to identify, extract and utilize such a signal.

## Standards of the development

Author develops the project since November 2022. During this time, the code base has grown significantly, and the architecture has become more complex. To prevent the code from turning into a patchwork quilt that is increasingly difficult to develop and maintain, the author adheres to the following rules in developing:

1. There is no or almost no technical debt: new functionality is seamlessly integrated into the project, and the integration itself is worked out immediately from start to finish, nothing is left “for later.” Sometimes this results in large-scale refactoring.

2. The code is full of checks. For example, it is necessary to guarantee that there are no gaps in the array, that the indices of two dataframes are equal, that the array has a certain dimension, etc. All this is explicitly checked. If the check fails, an exception is thrown, accompanied by some debugging information. Otherwise, such errors, not immediately identified, may remain undetected for a long time since the code will be executable in some cases. This makes development meaningless: for example, what is the use of a model if, during its training, the features and target features do not correspond to each other? Therefore, a lot of attention is paid to testing variables. In addition, this allows you to speed up debugging and searching for bugs in the future since you don’t have to re-check what has already been tested and works.

3. Generally, arguments of functions (methods) are positional, and named ones are used only in rare reasonable cases. This is necessary so that the arguments do not have default values, and each time a function (method) is called, the values passed to it are explicitly specified in the code. Readability improves and the likelihood of bugs decreases.

4. Object-oriented style: inheritance, encapsulation, and polymorphism are actively used. When writing code, attention is paid to its versatility and possible further reuse. There is no or almost no repeat code. Even one or two lines are wrapped into a function if used repeatedly.

5. Compliance with the hierarchy of abstraction levels. Code is sometimes separated into distinct functions just for this purpose, even if used only once. The same goes for classes: sometimes you can do without a base class, but it is still defined if there is a certain group of entities that are similar to each other. For example, there are metrics and loss functions. In the first case, the base class implements methods common to all metrics, so it is necessary. In the second case, the base class only inherits `_Loss` and does nothing else. However, this base class is necessary for hierarchy, and for another purpose, which will be discussed below. In this way, something like a generalized hierarchy is achieved: horizontal interaction between modules occurs, if possible, at the same level.

6. Functions (methods) with complex, non-trivial or not obvious functionality from the name are always accompanied by a detailed description, sometimes with examples. There is no need to re-immerse in the old code: with the help of description, the function algorithm is quickly remembered. The same goes for comments in the code: if the code is not self-explanatory, comments are added.

7. Code formatting is respected. The style is close to the "black" formatter.

8. Data type hints are used throughout. Not only with function arguments, but also directly in the code. For example, a list of strings is declared like this: `a: Iterable[str] = []`. This does not slow down writing code, but, on the contrary, significantly speeds it up: the IDE always knows the data type and provides the corresponding drop-down list. You can find out what type a variable is by simply hovering over it rather than tracking it through code. Debugging and maintaining time is minimised. The base classes discussed above are well suited for specifying a data type with them - this is their another important role.

9. The principle of zero tolerance towards crutches in the code because they cause errors and lead to accumulating technical debt is supported. There must not be any “hard code” left in the accomplished code.

## Pipeline

The research consists of several consequent stages, each of which solves a specific problem: feature extraction, feature selection, choosing the target feature, model training, model inference (obtaining data with it), etc. The actual research component is that the correct solution is not known in advance. It just has to be found - at the start there is only a formulated task. Each stage involves testing several different approaches and hundreds of ML experiments.

### Script for launching

To eliminate errors (and just for convenience), a shell script has been developed that performs launches. The names of the blocks to be run are passed through the argument. Firstly, this ensures that the blocks are launched in the correct sequence. Secondly, the correctness of the arguments passed to python scripts is guaranteed: once developed and debugged, they will not be accidentally misused. Thirdly, such a shell pipeline is suitable as a comprehensive top-level description of the whole project, something like a contents.

### Feature extraction

Source data - date-time, OHLC and volume. Features - all kinds of statistics and derivatives. Some ideas are basic and obvious (for example, diff and rolling mean), some are gleaned from open TSA repositories (e.g., tsfresh), and there are also proprietary features. Date and time are converted through cyclic (trigonometric) functions because they are cyclic natively. Attention is paid to preventing leaks from the future datapoints.

There is a base class `Calculator` and feature classes: one feature - one class.

For primary feature extraction, an arbitrary small slice of data is selected (specified in the configuration). During the primary extraction, many features are immediately eliminated because they do not meet the restrictions (too high correlation, too low variance, etc.). Those that remain are selected using RFE or RFECV (both algorithms are integrated into the project). The extraction and selection information is written to a file  (a table) and then used to extract features from the rest data. Based on the table, a pipeline is formed, providing deterministic and optimized extraction: with the least amount of auxiliary calculations, only what was obtained during the initial extraction and selection is extracted. By auxiliary calculations, we mean the calculation of intermediate features, which themselves were rejected during selection, but the required features are recursively extracted from them.

The data flow scheme is available [here](https://github.com/AndreyRysin/market-research-demo/blob/main/docs/data_flow.svg) (in `docs`).

The performance of selection algorithms was improved by splitting the selection process into iterations: features are rejected in small portions. This selection is repeated several times, and the results are combined in a certain way.

### Models training

A proprietary framework is used to train models. It is simple and functional: for each experiment, it saves metadata such as the training log, configuration, model architecture, means and standard deviations of features (if normalization is applied). Also, it saves the checkpoints. Models are saved in the inference-ready form: torchscript and onnx.

## Using open sources

In order not to reinvent the wheel and not miss interesting ideas, the author reads Medium, and when he comes across something valuable and worth studying, he turns to the original source on Arxiv.

This is how an effective encoder architecture was found. That article on Arxiv is accompanied by the source code (repository on Github), so full implementation was not required: it was enough to integrate the code into the project.

## Computational efficiency

Available hardware: 24 cores, 64 GB RAM, RTX2060s 8 GB.

Using kernels effectively is simple: initially, the data is natively divided into relatively small parts, and all that need is to process them in several parallel processes with `Pool`.

Dealing with memory is more challengeable. Data sometimes grows to enormous width. This becomes a problem when you need to cache the entire amount of data, but it is impossible to do this due to lack of memory. The problem is solved algorithmically at the cost of more computing time.

For example, to calculate the standard deviation of an entire array, it is necessary to iterate the data divided into parts twice: the first time to calculate the means, the second time to calculate the variances. When calculating the variance for each part of the array, the mean of the entire array (preliminarily calculated in the first pass) is used. This is done so because the goal is to calculate the standard deviation for the entire array, and not, say, the average of the standard deviations calculated for individual parts of the array. In this example, the amount of calculations due to the iterative nature of processing has not changed. Only disk reads doubled.

In another example, the amount of calculations doubles. The goal here is to save the disk: there is an experiment in which you need to write several tens of gigabytes of intermediate data to the disk, and this experiment needs to be carried out many times. The mean and the standard deviation are calculated, as in the previous example. However, the difference is that the array for which these statistics are calculated is obtained as a result of model inference. Therefore, not just reading from the disk occurs twice, but reading the source data from the disk plus processing it with a model. To save computational time and not do inference twice, one could store the intermediate data on disk and then read it a second time. But the time costs in absolute terms turned out to be such that disk resource is a priority: a couple of dozen experiments lasting 20 minutes versus 8 minutes are not worth a lost disk.

The third example is the calculation of the rolling correlation. There is an array of sizes approximately 7000 x 10000. It is necessary to calculate the rolling correlation over all possible combinations of columns over a window of size 100 with a step of 1, which is equivalent to calculating a regular (non-rolling) correlation for 6900 arrays of size 100 x 10000. Thus, the correlation matrix will have the size 10000 x 10000 x 6900 which is about 2.5 TB for float32. The entire correlation matrix is needed since the task is the feature selection (removing correlated ones). That is, after each iteration of removal, it is necessary to evaluate the remaining features in their entirety. The use of sparse matrices helped solve the problem. Since the interest is not the correlation value itself, but only the fact that the threshold is exceeded, the correlation values are not cached, but only a binary matrix is composed. Which, in turn, is converted to sparse and stored compactly in memory. The sparse matrix is stored and processed in parts. The size of the parts is selected so that each of the parts separately, being converted from a sparse matrix to a regular matrix, would fit into memory.

It is worth noting that in the rolling correlation problem, we also had to solve the problem of accelerating calculations in general. The final solution is about four orders of magnitude faster than the basic `pandas.DataFrame.rolling(win).corr()`. A speedup of two orders of magnitude is achieved by caching the denominator: a vector multiplied by itself only needs to be calculated once, rather than repeating this calculation N times for each column. Two more orders of magnitude - due to the transfer of calculations to the GPU (the code was rewritten in torch). As a result, what was estimated by the tqdm utility in days began to be computed in minutes.

## Tricks

1. Configuration is a regular class that itself can check values for correctness (for example, the left border of an interval should not be greater than the right); saves the configuration on disk (pickle, json, txt); checks that the parameters that need to be overridden have actually been overridden. All these functions help to fix bugs and generally save time: usually, an exception is raised at the very beginning of execution, and it immediately becomes clear where to look for the problem - the error message directly points to it.

2. `FeatureFrame` is a class inherited from `pandas.DataFrame`, which implements some additional methods. It’s worth honestly noting that a regular wrapper class would have done just as well, so the decision is controversial. Nevertheless, it turned out to be very convenient, it looks neater in the code, and it was just interesting to experiment.

## Further development

At the time of writing, in December 2023, the study is 70-80% complete. It remains to develop one model, and then move on to testing.

### Testing

Testing consists of two parts:

1. Calculation. It is necessary to develop a pipeline that takes input data (date-time, OHLC and volume) and returns a forecast. All the processes - feature extraction, model inference, etc. - are performed internally. Attention here is paid to the transparency of data flow: it is necessary to ensure that there are no leaks (even the most implicit ones) from the future.

2. Analysis. It is necessary to examine the obtained result using various metrics. Based on the results, conclude whether there are periods of predictable markets.

### Development of a trading strategy

If periods of predictable markets exist, then a trading strategy should be developed.

Usually, the development of a trading strategy means an attempt to identify some patterns of the market (intuitively or using the simplest mathematical apparatus), learn to recognize them and make a forecast based on them. That is, to do the work manually that models do in a given project.

In this project, developing a trading strategy means working with the models’ responses. The forecast is calculated automatically, and the task comes down to evaluate how much you can trust this forecast.
