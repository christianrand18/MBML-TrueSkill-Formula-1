> Act as an expert Data Scientist and Python developer. I am building a Bayesian Probabilistic Graphical Model (PGM) based on TrueSkill to evaluate Formula 1 driver and constructor skills. 
> 
> I have the Kaggle Formula 1 dataset stored in a local `data/` folder. I need a robust Python script using `pandas` to merge, clean, and transform these specific CSV files into a single, modeling-ready DataFrame.
> 
> **Here is the schema of the relevant files:**
> * `circuits.csv`: circuitId, circuitRef, name, location, country, lat, lng, alt, url
> * `constructors.csv`: constructorId, constructorRef, name, nationality, url
> * `drivers.csv`: driverId, driverRef, number, code, forename, surname, dob, nationality, url
> * `pit_stops.csv`: raceId, driverId, stop, lap, time, duration, milliseconds
> * `races.csv`: raceId, year, round, circuitId, name, date, time, url, fp1_date, ...
> * `results.csv`: resultId, raceId, driverId, constructorId, number, grid, position, positionText, positionOrder, points, laps, time, milliseconds, fastestLap, rank, fastestLapTime, fastestLapSpeed, statusId
> 
> **Please write a script that performs the following exact steps:**
> 
> 1. **Data Loading & Null Handling:** Load the necessary files. The Kaggle dataset uses the string `\N` to represent missing values. Replace all instances of `\N` with `np.nan` globally upon loading.
> 2. **Filter by Year:** The `pit_stops.csv` data only starts in 2011. Merge `results.csv` with `races.csv` (on `raceId`), and filter the dataset to only include races from the year 2011 onwards.
> 3. **Process Strategy (Pit Stops):** >     * Group `pit_stops.csv` by `raceId` and `driverId`.
>     * Calculate the total pit stop duration per driver per race (ensure the `milliseconds` column is numeric and sum it).
>     * Count the total number of pit stops per driver per race.
>     * Merge these aggregated strategy features back into the main results dataframe. Fill NaNs with appropriate values (e.g., 0 pit stops if they DNF'd on lap 1).
> 4. **Merge Identifiers and Weather Hooks:** >     * Merge the main dataframe with `circuits.csv` (on `circuitId`) to bring in `lat` and `lng`. Ensure `date` from `races.csv` is preserved as a standard datetime object. **This is strictly to prepare for a future weather API integration.**
>     * Ensure `driverId` and `constructorId` are preserved as integers.
> 5. **Feature Selection:** Drop all unnecessary metadata, URLs, strings, and redundant columns. The final output DataFrame should ONLY contain:
>     * `raceId`, `year`, `date`, `circuitId`, `lat`, `lng` (Environmental/Temporal context)
>     * `driverId`, `constructorId` (Latent Skill Identifiers)
>     * `grid`, `total_pit_duration_ms`, `num_pit_stops` (Observed Inputs)
>     * `positionOrder`, `statusId` (Observed Outputs)
> 6. **Final Cleanup:** Ensure the target variable `positionOrder` and inputs like `grid` are numeric (integers/floats). Drop any rows where critical modeling data (`positionOrder`, `grid`) is entirely missing.
> 
> Please provide the full, well-commented Python code to output this final `f1_model_ready.csv` file.

