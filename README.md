# Late Game Shot Optimization in the NBA

This project uses a neural network-based win probability model to inform optimal shot selection in the NBA. In this work, I sought to examine whether there were any advantages to be had by rejecting the now conventional wisdom of 3 > 2 when it comes to shot optimization in the NBA. The crux of my hypothesis was that while yes, 3 > 2, and a 3 converted 34% of the time still has an expected value higher than a 2 at 50% of the time, the higher conversion rate of the 2 could be preferential to teams when leading close late in the game.

To attempt to quantify this, I created a win probability model from scratch using play-by-play data from the R package hoopR collected from 2016 through 2025. I settled on a neural net architecture and fed it features very similar to those used by the inpredictable win probability model. The model ended up achieving an overall accuracy of 76.6% and a Brier score of 0.154. Breakdowns of those scores by quarter can be seen below:

<p align="center">
  <img src="plots/brier_by_quarter.png" width="45%" />
  <img src="plots/accuracy_by_quarter.png" width="45%" />
</p>

After completing the model, I found the mean win probability for the home team given time and score (and assuming the spread was 0) and visualized the results. Additionally, I refined this to only include the last 2 minutes and scores within 9 points in either direction. These can be seen below:

<p align="center">
  <img src="plots/wp_heatmap_full.png" width="80%" />
</p>

<p align="center">
  <img src="plots/wp_heatmap_clutch.png" width="80%" />
</p>

After this, I used the same play-by-play data to find the league average values for 3- and 2-point percentage, as well as rebound rates given both types of shots. Finally, using this data I calculated the change in expected winning percentage given what type of shot the home team attempted (whether that was a 3 or 2). To make the results more interpretable, I used a threshold of ±1.5% in eWP to decide whether a team should shoot a 3 or 2 given time and score (again, assuming a 0-point pregame spread). These results can be seen below:

<p align="center">
  <img src="plots/shot_type_recommendation.png" width="100%" />
</p>

The findings suggest teams should generally shoot 2s when the home team margin is between -1 and +1 under 45 seconds (Zone 1). Additionally, a two is considered more valuable when down -2 with time remaining roughly between 0:50 and 0:24. Conversely, when the margin is between -2 and -5 under roughly one full shot clock (24 seconds), teams should be shooting 3s (Zone 2). The model also suggests there could be value to be gained by shooting 3s when at a greater deficit with more time left, and this logically tracks with the idea of catching up on the scoreboard quickly.

## Usage

All datasets can be found in the `datasets` folder. Additionally, the `get_pbp.R` file can pull this data from hoopR and clean it. 

`neural_net.R` will train the neural net model and save it to the `models` folder. 
`nn_testing.R` will create the plots seen above for the win probability model.
`shot_values.R` parses the play-by-play data for league average shot values. 

And finally, `eWP.R` calculates expected winning percentage added after taking a 3. 

## Further work, caveats, etc.

The biggest caveat here is that these results are directly derived from my win probability model. That means that if the model is poorly calibrated or inaccurate, the results will be off. I feel fairly confident in the model itself; however, there are definitely things to iron out in the future. 

One other caveat is that the eWPA calculations use league average shooting and rebounding splits. The KD Warriors, for example, would likely have a different optimal late game shot strategy than a lottery team.

In the future, I would like to deploy both the win probability model and the eWPA model as web apps for public use.