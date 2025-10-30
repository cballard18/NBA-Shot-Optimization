library(tidyverse)
library(hoopR)
library(arrow)

#load data in from hoopR
#seasons <- c(2016:2025)
#df <- hoopR::load_nba_pbp(seasons = seasons)

#save raw pbp
#write_parquet(df, "datasets/raw_pbp.parquet")

#read raw pbp
setwd("/Users/charliepersonal/Desktop/Sports/NBA/Win Prob")
df <- read_parquet("datasets/raw_pbp.parquet")

#fix one cell for a matt bonner foul in 2016
df$team_id[90503] <- 24
#fix score error in pistons hawks game in dec 2017
df$home_score[1603184] <- 26
df$home_score[1603185] <- 26

#using reg, play in, play off data. to remove play off filter only season_type == 2

clean_df <- df %>%
  filter(home_team_id < 100, away_team_id < 100) %>%
  mutate(
    end_game_seconds_remaining = if_else(
      text == "End of the 4th Quarter" | type_text == "End Game",
      0, end_game_seconds_remaining
    )
  ) %>%
  fill(end_game_seconds_remaining, .direction = "down") %>%
  mutate(
    home_margin = home_score - away_score,

    #cache foul detection
    is_foul = str_detect(type_text, regex("Foul", ignore_case = TRUE)) &
      !str_detect(type_text, regex("Technical Foul|No Foul", ignore_case = TRUE)),
    home_foul = as.integer(is_foul & team_id == home_team_id),
    away_foul = as.integer(is_foul & team_id == away_team_id),
    foul = as.integer(home_foul == 1 | away_foul == 1),
    fg3a = as.integer(str_detect(text, regex("three", ignore_case = TRUE))),
    fg3m = as.integer(fg3a == 1 & scoring_play),
    fta = as.integer(str_detect(text, regex("free throw", ignore_case = TRUE))),
    ftam = as.integer(fta == 1 & scoring_play),
    fg2a = as.integer(shooting_play & fg3a == 0 & fta == 0),
    fg2m = as.integer(fg2a == 1 & scoring_play),
    fgm = as.integer(fg2m == 1 | fg3m == 1),
    tov = as.integer(
      str_detect(text, regex("turnover", ignore_case = TRUE)) |
        str_detect(type_text, regex("turnover", ignore_case = TRUE))
    ),
    dreb = as.integer(
      str_detect(text, regex("Defensive Rebound", ignore_case = TRUE)) |
        str_detect(type_text, regex("Defensive Rebound", ignore_case = TRUE))
    ),
    oreb = as.integer(
      str_detect(text, regex("Offensive Rebound", ignore_case = TRUE)) |
        str_detect(type_text, regex("Offensive Rebound", ignore_case = TRUE))
    ),
    home_foul_total = cumsum(home_foul),
    away_foul_total = cumsum(away_foul),
    home_win = if_else(max(home_score) > max(away_score), 1L, 0L),
    lead_dreb = lead(dreb),
    lead_oreb = lead(oreb),
    action_team = team_id,
    home_team_action = as.integer(action_team == home_team_id),
    pos_team = case_when(
      home_team_action == 1 & foul == 1 ~ away_team_id,
      home_team_action == 0 & foul == 1 ~ home_team_id,
      TRUE ~ action_team
    ),
    home_team_pos = as.integer(pos_team == home_team_id),
    .by = game_id
  )

rm(df, seasons)
gc()

write_parquet(clean_df, "datasets/clean_pbp.parquet")

filtered_clean_df <- clean_df |>
  select(game_id, end_game_seconds_remaining, home_margin, home_team_spread, home_win, home_team_pos)

arrow::write_parquet(filtered_clean_df, "datasets/filtered_pbp.parquet")
