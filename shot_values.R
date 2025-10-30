library(tidyverse)
library(arrow)

df <- read_parquet("datasets/clean_pbp.parquet")

#league average shooting splits
shots <- df |>
  filter(shooting_play == TRUE) |>
  mutate(
    fg3a = ifelse(grepl("three", text, ignore.case = TRUE), 1, 0),
    fg3m = ifelse(fg3a == 1 & scoring_play == TRUE, 1, 0),
    fta = ifelse(grepl("free throw", text, ignore.case = TRUE), 1, 0),
    ftam = ifelse(fta == 1 & scoring_play == TRUE, 1, 0),
    fg2a = ifelse(shooting_play == TRUE & fg3a == 0 & fta == 0, 1, 0),
    fg2m = ifelse(fg2a == 1 & scoring_play == TRUE, 1, 0)
  )
shots |>
  summarise(
    fg3p = mean(fg3m[fg3a == 1]),
    fg2p = mean(fg2m[fg2a == 1]),
    ftp = mean(ftam[fta == 1])
  )
fg3_mean = 0.358
fg2_mean = 0.517
ft_mean = 0.773

#reb chances
rebs <- df |>
  mutate(
    fg3a = ifelse(grepl("three", text, ignore.case = TRUE), 1, 0),
    fg3m = ifelse(fg3a == 1 & scoring_play == TRUE, 1, 0),
    fta = ifelse(grepl("free throw", text, ignore.case = TRUE), 1, 0),
    ftam = ifelse(fta == 1 & scoring_play == TRUE, 1, 0),
    fg2a = ifelse(shooting_play == TRUE & fg3a == 0 & fta == 0, 1, 0),
    fg2m = ifelse(fg2a == 1 & scoring_play == TRUE, 1, 0)
  ) |>
  mutate(
    dreb = ifelse(
      grepl("Defensive Rebound", text, ignore.case = TRUE) |
        grepl("Defensive Rebound", type_text, ignore.case = TRUE), 1, 0
    ),
    oreb = ifelse(
      grepl("Offensive Rebound", text, ignore.case = TRUE) |
        grepl("Offensive Rebound", type_text, ignore.case = TRUE), 1, 0
    )
  ) |>
  group_by(game_id) |>
  mutate(
    lead_dreb = lead(dreb, n = 1L),
    lead_oreb = lead(oreb, n = 1L)
  )

rebs |>
  ungroup() |>
  summarise(
    fg3_dreb = mean(lead_dreb[fg3a == 1 & fg3m == 0]),
    fg3_oreb = mean(lead_oreb[fg3a == 1 & fg3m == 0])
  )

fg3_dreb_pct = 0.754
fg3_oreb_pct = 0.243

rebs |>
  ungroup() |>
  summarise(
    fg2_dreb = mean(lead_dreb[fg2a == 1 & fg2m == 0], na.rm = TRUE),
    fg2_oreb = mean(lead_oreb[fg2a == 1 & fg2m == 0], na.rm = TRUE)
  )

#doesnt quite add to 1 due to data errors
fg2_dreb_pct = 0.684
fg2_oreb_pct = 0.303