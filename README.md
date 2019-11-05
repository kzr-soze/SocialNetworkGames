# SocialNetworkGames
Currently uniform exponential games on a demo social network (see demo() function in main.py) by running `python main.py game_type`
from the command line, where `game_type` is one of the currently functional types.

 | `game_type` | Description|
 | ----------------- | -----------|
 | `collab` | Collaborative game, where each player chooses `k` players to play with, and plays one game with each as well as any players which select it as one of their `k` players.|
 | `partner` | Willing partnership, each player plays at most `k` games, and only with partners who agree to play as one of their `k` games|
 | `dictator` | Random dictatorship, each round players are randomly ordered and earlier players select at most `k` neighbors to play with without their input. Each player plays at most `k` games per round |
