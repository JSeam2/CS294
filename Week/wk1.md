# Week 1 Notes
## Unsupervised Learning In a Nutshell
- We want to learn from **unlabelled data***
    - **Generative Models***: recreate raw data distribution
    - **Self-supervised learning**: "Puzzle" tasks that require semantic understanding

- Cake analogy
    - **Reinforcement Learning is the "Cherry"**: Bot gets scalar reward once in a
      while.

      We get few bits info for some samples

    - **Supervised Learning is the "Icing"**: Predicts human supplied data

      We get 10-10,000 bits of info per sample

    - **Unsupervised/Predictive Learning is the "Cake"**: Predicts any parts of
      input per observed part. Prediction of future frames.

      We get millions bits of info per sample

- Intelligence is about **compression** and **pattern finding**
    - Finding all patterns = short description of raw data (low Kolgorov
      Complexity)

    - Shortest code length = optimal inference (Solomonoff Induction)

    - Extensible to optimal decision making agents (AIXI)

- Applications of Deep Unsupervised Learning
    - Generation of novel data
    - Compression
    - Improve downstream tasks
    - Flexible building blocks

