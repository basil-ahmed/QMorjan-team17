# NYUAD Hackathon for Social Good in the Arab World: Focusing on Quantum Computing (QC)

# QMarjan: Coral Reef Restoration Optimization

## Introduction

QMarjan is a pioneering project combining classical computer vision (CV) techniques and quantum computing to optimize coral reef repopulation strategies, starting with the United Arab Emirates (UAE). By leveraging satellite imagery and quantum algorithms, QMarjan aims to address the global challenge of coral reef degradation.


https://github.com/basil-ahmed/QMorjan-team17/assets/90772853/2f5a1e15-c5d4-490a-91b2-c51f10dfacda



## Background

Coral reefs are crucial for marine life and human economies, but are at risk due to climate change. QMarjan's approach uses advanced algorithms to identify the best locations for coral restoration and utilizes quantum computing to determine optimal repopulation strategies.

## Classical Computer Vision Model

### Overview

The Classical CV Model utilizes unsupervised learning with Gaussian Mixture Models (GMM) for real-time detection of coral reefs from satellite imagery.

### Dependencies

- Python 3.8+
- OpenCV
- scikit-learn
- numpy

### Installation

```bash
git clone https://github.com/your_github_username/qmarjan.git
```

### Usage

```bash
python coral_detection.py --image_path /path/to/satellite/image
```

### Output

The model outputs an image highlighting detected coral reefs and a CSV file with coordinates of detected areas.

## Quantum Computing Model

### Overview

The Quantum Model employs Quantum Annealing to solve the Set Cover Problem for determining optimal coral repopulation points.

### Dependencies

- qBraid
- D-Wave Ocean SDK

### Installation

Ensure you have access to a quantum computing service like D-Wave through qBraid.

```bash
git clone https://github.com/your_github_username/qmarjan.git
```

### Usage

The model requires an input graph representation of detected coral reefs from the Classical CV Model.

```bash
python bitmap_things.ipynb.py --graph_path /path/to/coral_graph
```

### Output

The algorithm provides a set of points representing the ideal locations for coral repopulation.

## Data Visualization

All the images (results) generated are present in the repository.

## Roadmap

- 6 months: Product validation with UAE MOCCAE and the "Dubai Reef" project.
- 3+ years: Scale to 15+ countries with separate data management systems and bleaching forecasting.

## Contributing

We welcome contributions from the community.

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.

## Acknowledgments

A special thanks to the mentors and students involved in the QMarjan project, including those from NYUAD, MIT, and other institutions.
