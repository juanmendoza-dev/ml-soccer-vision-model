# References

## Vision
- [roboflow/sports](https://github.com/roboflow/sports) — YOLOv8 player/ball/pitch detection, team clustering, top-down minimap. Starting point for the vision pipeline.
- [SoccerNet/sn-gamestate](https://github.com/SoccerNet/sn-gamestate) — Research-grade broadcast tracking with jersey-number identification. Use for player identity.

## Prediction
- [SoccerAI](https://alessioarcara.github.io/SoccerAI/) — Temporal GNN predicting shot probability from game state with player stats. Closest existing project.
- [UnravelSports/unravelsports](https://github.com/UnravelSports/unravelsports) — Converts tracking data from many providers into graphs for GNN training.
- [USSoccerFederation/ussf_ssac_23_soccer_gnn](https://github.com/USSoccerFederation/ussf_ssac_23_soccer_gnn) — GNN for counterattack success with data from 632 pro matches. Adaptable to shots.
- [hyunsungkim-ds/defcon](https://github.com/hyunsungkim-ds/defcon) — GNN defensive valuation; includes a ready xG training CSV.
- [LaurieOnTracking](https://github.com/Friends-of-Tracking-Data-FoTD/LaurieOnTracking) — Pitch control and EPV tutorials on Metrica data.

## Papers
- [Making Offensive Play Predictable (Stats Perform)](https://www.statsperform.com/wp-content/uploads/2021/04/Making-Offensive-Play-Predictable.pdf) — Industry GNN predicting shots in the next 10 s.
- [Goka et al. 2023, Sensors](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10181557/) — Bipartite-graph GCRNN for shot prediction with uncertainty.
- [Bekkers & Sahasrabudhe, GNN counterattacks](https://arxiv.org/abs/2411.17450) — Paper behind the USSF repo and unravelsports.
