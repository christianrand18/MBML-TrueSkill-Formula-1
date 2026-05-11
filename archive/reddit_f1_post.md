# Whole-History Ratings of Drivers & Constructors (as of the end of the 2023 season)
**Author:** Kezyma
**Subreddit:** r/formula1

I've posted a few different ratings and stats before, but I recently reimplemented some of the rating systems I use and decided to use F1 data for testing purposes. I thought the results might be interesting to share, so here they are!

## Background

The Whole-History rating system is an improvement on the Elo rating system (commonly used for chess and online games). One of the main improvements is that while Elo only updates ratings in one direction, gaining or losing with each subsequent game played, Whole-History works both directions in time, retroactively updating past ratings based on new results, allowing for a better comparison of players across different time periods and mitigating rating inflation due to consistent wins against notably weak opposition. The original paper for the system can be found here.

The data I used was from this Kaggle dataset of all race results in F1 history. I excluded all the early Indy 500 results.

These ratings are all generated with default parameters and the same system could produce more accurate ratings with some hyperparameter optimisation, but no amount of tuning should move anyone more than a few positions in any direction.

## Drivers Ratings

For each race, a series of 1v1 games are played between each driver. DNFs are excluded. 

The final ratings at the end of 2023 were as follows;
*(2023 Driver Ratings)*

Below are the top 100 all-time peak ratings. One thing to note here is that these ratings include car performance at the time as much as they do the driver and are very much relative to the grid at the time. Finishing on the podium every weekend because you have the best car is going to give you a better rating than winning the championship in a competitive year where a bunch of different teams get wins.

Another interesting point is that there's a built-in recency bias. If a driver is currently on top, or retired immediately at their peak, their rating wont have retroactively lowered to a more 'accurate' final approximation. This effect is fairly minimal over a long enough time period however.

Please ignore the messed up names, it was an encoding issue and I didn't feel like going through the list fixing the names.
*(Top 100 All-Time Driver Peak Ratings)*

## Constructor Ratings

For the constructor ratings, each race is a series of 1v1 games between the best finishing car for each constructor. DNFs are again excluded.

The final ratings at the end of 2023 are as follows;
*(2023 Constructor Ratings)*

And the all-time peaks are below. I didn't bother trying to combine or separate any constructors and left them as they were in the original dataset, so some may technically appear twice.
*(All-Time Peak Constructor Ratings)*

## Teammate Ratings

These are the most interesting in my opinion, and can't realistically be done without the retroactive updates of WHR that smoothen out the rating change over time.

Each race is a 1v1 between teammates, ignoring other teams entirely. DNFs again excluded.

This is the closest you can really get to rating drivers in equal cars, although it has a few flaws, in that if you haven't had multiple teammates, or you've only had really strong or really weak teammates, the ratings get a bit skewed.

The final teammate ratings for 2023 were as follows;
*(2023 Teammate Ratings)*

And the top 100 all-time teammate ratings. Lots of older drivers do well here since the standards weren't as high. The same sort of bias appears here too, with a dominant season against a poor teammate being much more rewarding than a close season against a high quality teammate.
*(Top 100 All-Time Peak Teammate Ratings)*

I might try optimising the parameters to get more accurate ratings in future, and obviously driver/constructor quality can't be measured by a single overall number and comparing across time gets even messier, so try not to take these too seriously, they were just testing data after all!

### Edit:
Below is a plot of the rating history of every F1 world champion. It's quite big so will need to be opened on desktop and probably zoomed in a bit to navigate!
*(Whole-History Ratings of Drivers & Constructors plot)*
