# ==========================================
# STEP 1: IMPORT LIBRARIES
# ==========================================
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import numpy as np

sns.set(style="whitegrid")

# ==========================================
# STEP 2: LOAD DATASETS (CSV FILES)
# ==========================================
players_df = pd.read_csv("data/players_data.csv")
goals_df = pd.read_csv("data/goals_data.csv")
attacking_df = pd.read_csv("data/attacking_data.csv")
distribution_df = pd.read_csv("data/distribution_data.csv")
teams_df = pd.read_csv("data/teams_data.csv")  # <-- NEW

print("✅ Files loaded successfully!\n")
print("Players:", players_df.shape)
print("Goals:", goals_df.shape)
print("Attacking:", attacking_df.shape)
print("Distribution:", distribution_df.shape)
print("Teams:", teams_df.shape)

# ==========================================
# STEP 3: CLEAN / PREP DATA
# ==========================================
players_df = players_df[["id_player", "player_name", "nationality", "field_position", "age", "id_team"]]
goals_df = goals_df.fillna(0)
attacking_df = attacking_df.fillna(0)
distribution_df = distribution_df.fillna(0)

# ==========================================
# STEP 4: MERGE DATASETS
# ==========================================
# rename team_id and team to match id_team and team_name
teams_df = teams_df.rename(columns={"team_id": "id_team", "team": "team_name"})

merged_df = (
    players_df
    .merge(goals_df, on="id_player", how="left")
    .merge(attacking_df, on="id_player", how="left")
    .merge(distribution_df, on="id_player", how="left")
    .merge(teams_df[["id_team", "team_name", "country"]], on="id_team", how="left")
)

print("\n✅ Merged dataset shape:", merged_df.shape)
print("Columns:", merged_df.columns.tolist())

# ==========================================
# STEP 5: BASIC DATA EXPLORATION
# ==========================================
print("\n--- BASIC DATA INFO ---")
print(merged_df.info())
print("\n--- SAMPLE DATA ---")
print(merged_df.head())

# ==========================================
# STEP 6: SAVE TO SQLITE DATABASE
# ==========================================
conn = sqlite3.connect("soccer_project.db")
merged_df.to_sql("soccer_data", conn, if_exists="replace", index=False)
conn.close()
print("\n✅ Data successfully saved to soccer_project.db (table: soccer_data)")

# ==========================================
# STEP 7: EXPLORATORY DATA ANALYSIS (EDA)
# ==========================================
top_scorers = merged_df.sort_values(by="goals", ascending=False).head(10)

plt.figure(figsize=(10,5))
sns.barplot(x="goals", y="player_name", data=top_scorers, hue="player_name", palette="viridis", legend=False)
plt.title("Top 10 Goal Scorers")
plt.xlabel("Goals")
plt.ylabel("Player")
plt.tight_layout()
plt.show()

# ==========================================
# STEP 8: PREDICTIVE ANALYSIS (Next Season Forecasts)
# ==========================================

print("\n🔮 Running simple predictive analysis for next season...")

# ---- Predict next year's top 5 goal scorers ----
goal_data = merged_df[["player_name", "age", "goals"]].dropna()
X = goal_data[["age"]]
y = goal_data["goals"]

model_goals = LinearRegression()
model_goals.fit(X, y)
goal_data["predicted_goals_next_season"] = model_goals.predict(X) + np.random.normal(0, 2, len(X))
top_future_scorers = goal_data.sort_values(by="predicted_goals_next_season", ascending=False).head(5)

print("\n🏆 Predicted Top 5 Goal Scorers Next Season:")
print(top_future_scorers[["player_name", "predicted_goals_next_season"]])

# ---- Predict next year's top 5 assist makers ----
assist_data = attacking_df[["id_player", "assists"]].merge(players_df[["id_player", "player_name"]], on="id_player")
assist_data["predicted_assists_next_season"] = assist_data["assists"] * 1.1
top_assists = assist_data.sort_values(by="predicted_assists_next_season", ascending=False).head(5)

print("\n🎯 Predicted Top 5 Assist Makers Next Season:")
print(top_assists[["player_name", "predicted_assists_next_season"]])

# ---- Predict Champions League Top 5 Teams ----
team_goals = merged_df.groupby("team_name")["goals"].sum().reset_index()
team_goals["predicted_points"] = team_goals["goals"] * np.random.uniform(2.0, 3.0, len(team_goals))
top_teams_pred = team_goals.sort_values(by="predicted_points", ascending=False).head(5)

print("\n⚽ Predicted Champions League Standings (Top 5):")
for i, row in enumerate(top_teams_pred.itertuples(), 1):
    print(f"{i}. {row.team_name} - Predicted Points: {row.predicted_points:.1f}")

# ==========================================
# STEP 9: SAVE VISUALS
# ==========================================
plt.figure(figsize=(10,5))
sns.barplot(x="predicted_goals_next_season", y="player_name", data=top_future_scorers, palette="magma")
plt.title("Predicted Top 5 Goal Scorers - Next Season")
plt.tight_layout()
plt.savefig('/Users/azwer/Desktop/soccer_project/predicted_top_scorers.png')
plt.close()

plt.figure(figsize=(10,5))
sns.barplot(x="predicted_points", y="team_name", data=top_teams_pred, palette="Blues_r")
plt.title("Predicted Top 5 Teams - Champions League")
plt.tight_layout()
plt.savefig('/Users/azwer/Desktop/soccer_project/predicted_top_teams.png')
plt.close()

# ==========================================
# STEP 10: FINAL SUMMARY
# ==========================================
print("\n\n================= FINAL SOCCER REPORT =================\n")
print(f"Total Players Analyzed: {merged_df.shape[0]}")
print(f"Average Age: {merged_df['age'].mean():.1f} years")
print(f"Top Predicted Goal Scorer: {top_future_scorers.iloc[0]['player_name']}")
print(f"Top Predicted Assist Maker: {top_assists.iloc[0]['player_name']}")
print(f"Predicted Champions League Winner: {top_teams_pred.iloc[0]['team_name']}")
print("\n✅ Charts saved as:")
print(" - predicted_top_scorers.png")
print(" - predicted_top_teams.png")
print("\n✅ Soccer predictions successfully generated!\n")


