---
dataset_info:
  features:
  - name: problem
    dtype: string
  - name: solution
    dtype: string
  - name: answer
    dtype: string
  - name: subject
    dtype: string
  - name: level
    dtype: int64
  - name: unique_id
    dtype: string
  splits:
  - name: train
    num_bytes: 9803889
    num_examples: 12000
  - name: test
    num_bytes: 400274
    num_examples: 500
  download_size: 5333852
  dataset_size: 10204163
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
  - split: test
    path: data/test-*
---

# Hendrycks MATH Dataset

## Dataset Description

The MATH dataset is a collection of mathematics competition problems designed to evaluate mathematical reasoning and problem-solving capabilities in computational systems. Containing 12,500 high school competition-level mathematics problems, this dataset is notable for including detailed step-by-step solutions alongside each problem.

### Dataset Summary

The dataset consists of mathematics problems spanning multiple difficulty levels (1-5) and various mathematical subjects including:

- Prealgebra
- Algebra
- Number Theory
- Counting and Probability 
- Geometry
- Intermediate Algebra
- Precalculus

Each problem comes with:
- A complete problem statement
- A step-by-step solution
- A final answer
- Difficulty rating
- Subject classification

### Data Split

The dataset is divided into:
- Training set: 12,000
- Test set: 500 problems

## Dataset Creation

### Citation

```
@article{hendrycksmath2021,
    title={Measuring Mathematical Problem Solving With the MATH Dataset},
    author={Dan Hendrycks
    and Collin Burns
    and Saurav Kadavath
    and Akul Arora
    and Steven Basart
    and Eric Tang
    and Dawn Song
    and Jacob Steinhardt},
    journal={arXiv preprint arXiv:2103.03874},
    year={2021}
}
```

### Source Data

The problems originate from high school mathematics competitions, including competitions like the AMC 10, AMC 12, and AIME. These represent carefully curated, high-quality mathematical problems that test conceptual understanding and problem-solving abilities rather than just computational skills.

### Annotations

Each problem includes:
- Complete problem text in LaTeX format
- Detailed solution steps
- Final answer in a standardized format
- Subject category
- Difficulty level (1-5)

### Papers and References

For detailed information about the dataset and its evaluation, refer to "Measuring Mathematical Problem Solving With the MATH Dataset" presented at NeurIPS 2021. 

https://arxiv.org/pdf/2103.03874