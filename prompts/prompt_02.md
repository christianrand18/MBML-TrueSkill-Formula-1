> Act as a Senior Machine Learning Engineer and Bayesian Statistician. I am building a Model-Based Machine Learning pipeline to rate Formula 1 drivers and constructors using the `trueskill` Python package. 
>
> I have a preprocessed Pandas DataFrame containing chronologically sorted historical race data with the following columns: `raceId`, `year`, `date`, `circuitId`, `lat`, `lng`, `driverId`, `constructorId`, `grid`, `total_pit_duration_ms`, `num_pit_stops`, `positionOrder`, `statusId`.
>
> Please write a sophisticated, production-ready Python module to implement a Baseline TrueSkill F1 model. The codebase must be highly professional, intuitive, and clean.
>
> **Requirements for the Codebase:**
> 
> 1. **Object-Oriented Architecture:** Structure the code using classes (e.g., `F1TrueSkillEnvironment`, `RaceProcessor`, `SkillEvaluator`).
> 2. **Type Hinting & Docstrings:** Use standard Python `typing` and Google-style or NumPy-style docstrings for all classes and methods.
> 3. **Logging:** Use Python's built-in `logging` module instead of `print` statements to track the chronological processing of races.
> 4. **TrueSkill Logic for F1:**
>    * Initialize a global `trueskill.TrueSkill` environment.
>    * Maintain a dictionary tracking the `trueskill.Rating` objects for each unique `driverId` and `constructorId`.
>    * **The Match Setup:** In F1, a race is a "Free-for-All" among 20+ competitors. Formulate each competitor as a TrueSkill "team" consisting of two entities: `[driver_rating, constructor_rating]`. 
>    * **The Update Loop:** Iterate through the DataFrame group-by-group (grouped by `raceId` chronologically). For each race, extract the `positionOrder` to define the finishing ranks, pass the "teams" and ranks to `trueskill.rate()`, and update the global dictionaries with the new posterior ratings.
> 5. **Extensibility:** Design the class structure so that it is easy to swap out the `trueskill` package backend for a custom `Pyro` backend in the future (e.g., isolate the `update_skills()` method).
> 6. **Evaluation & Output:** Include a method to extract the final $\mu$ and $\sigma$ for all drivers and constructors into a cleanly sorted Pandas DataFrame so I can easily see the "All-Time Greats" according to the model.
>
> Please provide the complete Python script (`f1_trueskill_baseline.py`), it should print what is happening and all output files should be placed new folders you make. do so where relevant and name the folders what they should be.
