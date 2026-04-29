# DTU Model-based Machine Learning: Project Success Guide

This document outlines the essential requirements, success strategies, and pitfalls to avoid for the Model-based Machine Learning project at DTU, based on the provided course slides. 

## 1. Project Deliverables and Deadlines
To succeed, you must ensure you meet the exact format and deadline requirements:
* **Deadline:** The final delivery is on 15 May, completed in groups [cite: 76].
* **Deliverable 1:** A fully self-explanatory notebook (similar to the ones provided in class) [cite: 77].
* **Deliverable 2:** A 6-page report (which includes figures and tables) formatted in the double-column IEEE research paper template [cite: 77].

## 2. Choosing Your Project Focus
The project topic is free, and creativity is highly encouraged [cite: 17]. You should focus your project within one of these four main classes [cite: 25]:
* **Problem-driven:** Select a problem you care about, find relevant data, and formulate a Probabilistic Graphical Model (PGM) that fits the problem [cite: 26].
* **Data-driven:** Choose a rich dataset, identify an appropriate research question, and formulate a PGM to answer it [cite: 36].
* **Model-driven:** Combine several modeling ideas covered in lectures to create a new model, implement it in Pyro, and test it [cite: 47, 48].
* **Paper-driven:** Find a research paper featuring a strong probabilistic modeling approach, formulate it as a PGM and generative process, implement it in Pyro, and test it [cite: 60, 61].
* *Note: If you are ever in doubt about your direction, talk to the instructors [cite: 62].*

## 3. What to Focus On for Success (Best Practices)
To build a successful project, prioritize these implementation steps:
* **Start Simple & Iterate:** Begin with a simple model and incrementally make it more complex until you reach your idealized PGM [cite: 143]. 
* **Use Baselines:** Always try to have baseline models for comparison to evaluate your new model's performance [cite: 144].
* **Verify with Artificial Data:** Use ancestral sampling to generate artificial data. Run inference on this artificial data using Pyro to verify if the model can recover the true values or parameters. This is a crucial step to guarantee your model is correctly implemented and inference works [cite: 157, 158].
* **Test Multiple Inference Algorithms:** Try using different types of inference algorithms, specifically Variational Inference (VI) and Markov Chain Monte Carlo (MCMC), to verify they are doing the right thing and producing trustworthy results [cite: 133].
* **Careful Prior Selection:** Choosing priors is often more art than science [cite: 121, 130]. Expect a lot of trial and error, but follow established guidelines, such as the Stan Prior Choice Recommendations (https://github.com/stan-dev/stan/wiki/Prior-Choice-Recommendations) [cite: 131, 132].

## 4. What to Be Careful Of (Common Pitfalls)
Avoid these common mistakes and misconceptions during your modeling process:
* **Priors on Observed Variables:** There is no point in putting priors on observed variables that lack parents (unless you are performing imputation), because these variables are already given [cite: 83].
* **Blocking the Information Path:** Do not include observed variables that block the information path in your PGM [cite: 90]. 
    * *Example:* In the chain "Intelligence → Course grade → Recommendation letter", if the course grade is always observed, it blocks the path between intelligence and the recommendation letter [cite: 91]. 
    * This effectively breaks the system into two independent models ("Intelligence → Course grade" and "Course grade → Recommendation letter") [cite: 92]. Estimating them jointly is pointless since there is no flow of information between them [cite: 93].
* **Superficial Dependency Modeling:** Do not just think about *which* variables depend on others and their distribution types. You must carefully think about *how* to model those dependencies (e.g., how exactly to condition the parameters of a Beta distribution on another variable) [cite: 105, 106].
* **Discrete Latent Variables:** Be extremely careful when including discrete latent variables, as they require special treatment in Pyro [cite: 113].

## 5. Potential Project Themes & Inspiration
If you need inspiration, the slides provided examples of past or suggested project themes:
* **Social Networks:** Inferring group memberships using a Mixed Membership Stochastic Block Model (MMSBM) [cite: 168].
* **Sports Analysis:** Semi-parametric soccer analysis using Cox proportional hazards models and Gaussian processes [cite: 285, 286].
* **Recommender Systems:** Movie Lens dataset analysis inspired by Bishop's book [cite: 295, 296].
* **Spatial Count Models:** Analyzing youth pedestrian injury counts using Negative Binomial distributions with hierarchical modeling, spatial correlations, and neural networks [cite: 317, 327, 364, 365].
* **Traffic Accident Severity:** Modeling injury severity levels using ordered logit models, exploiting hierarchical data structures (individuals nested in vehicles, vehicles in events) [cite: 384, 385, 407].
