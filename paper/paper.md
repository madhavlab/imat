---
title: 'IMAT: Interactive Melody Annotation Tool'
tags:
  - Python
  - Machine Learning
  - Melody Annotation
  - Domain Adaptation
authors:
  - name: Kavya Ranjan Saxena
    orcid: 0000-0002-6590-2019
    affiliation: 1 
  - name: Vipul Arora
    affiliation: 1
affiliations:
 - name: Indian Institute of Technology (IIT) Kanpur, India
   index: 1
date: 5 May 2025
bibliography: paper.bib

---

# Summary
Estimating singing melody from polyphonic audio is a fundamental and important task in the music information retrieval field. There are many downstream applications of melody estimation, including music recommendation[[`@mr`]](https://ieeexplore.ieee.org/document/9414458), cover song identification[[`@cvi`]](https://ieeexplore.ieee.org/document/9747630), music generation[[`@mg`]](https://archives.ismir.net/ismir2020/paper/000146.pdf), and voice separation[[`@vs`]](https://ieeexplore.ieee.org/document/7178034). In order to achieve high performance in the downstream applications, the estimated melody should be highly accurate. There are both signal processing[[`@sp1`]](https://repositori-api.upf.edu/api/core/bitstreams/1864c4d1-2c39-4474-9578-4da95d30f391/content)[[`@sp2`]](https://ieeexplore.ieee.org/document/5431024), and machine learning-based[[`@ml1`]](https://archives.ismir.net/ismir2018/paper/000286.pdf)[[`@ml2`]](https://brianmcfee.net/papers/ismir2017_salience.pdf) algorithms for estimating the melody from the polyphonic audios. The drawback of these methods is that the estimated melody may be inaccurate. The inaccuracies can be caused by the presence of loud accompaniments, inherent noise in the audio, or model inaccuracies. The user must correct the inaccurately estimated melody to make it suitable for downstream applications. One way of correcting the melody is by manual annotation, which is a time-consuming and labor-intensive process. Another way could be using a machine learning algorithm that reduces the human labor and expedites the annotation process. In this work, we have developed IMAT, an interactive tool that uses our previously proposed model-agnostic machine-learning-based algorithm, i.e., active-meta-learning~\cite{saxena2024} that combines active-learning[[`@al`]](https://dl.acm.org/doi/pdf/10.1145/3472291) and meta-learning[[`@maml`]](https://proceedings.mlr.press/v70/finn17a/finn17a.pdf) to achieve the above-mentioned goal. When audio is given as an input to IMAT, the corresponding spectrogram is calculated and the algorithm uses active learning to identify the low-confidence time frames. These frames are available for the user to correct by manual annotation which can be aided by both visual and auditory feedback. The algorithm then adapts to these corrections using meta-learning, thus providing a more precise melody annotation of the entire audio. This process, referred to as adaptive annotation, allows users to achieve high annotation accuracy while significantly reducing the annotation time. 


# Statement of Need
