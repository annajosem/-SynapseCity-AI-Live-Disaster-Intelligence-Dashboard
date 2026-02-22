# synapsecity_dashboard_live.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import folium
from streamlit_folium import st_folium
import branca.colormap as cm
import requests

# -------------------------------
# 0. Risk Weights
# -------------------------------
alpha, beta, gamma = 0.5, 0.3, 0.2  # Flood, Population, Panic
w_pop, w_fear = 0.4, 0.6  # PanicScore components

# -------------------------------
# 1. Dataset for Irinjalakuda Zones (example coordinates)
# -------------------------------
data = {
    "Zone": [
        "Town Center", "Railway Station", "Market Area", "Hospital Area",
        "College Road", "Temple Road", "Residential North", "Residential South",
        "Riverbank Area", "Industrial Zone"
    ],
    "PopulationDensity": [300, 200, 400, 150, 180, 120, 250, 220, 80, 100],
    "DrainageEfficiency": [0.3, 0.5, 0.4, 0.8, 0.6, 0.7, 0.5, 0.4, 0.2, 0.6],  # 0-1
    "SoilPermeability": [0.2, 0.6, 0.4, 0.8, 0.5, 0.7, 0.6, 0.3, 0.2, 0.5],  # 0-1
    "Elevation": [5, 10, 8, 12, 9, 15, 11, 7, 4, 10]  # meters
}
df = pd.DataFrame(data)
df["Lat"] = [10.320, 10.325, 10.322, 10.318, 10.321, 10.319, 10.317, 10.315, 10.323, 10.316]
df["Lon"] = [76.222, 76.224, 76.226, 76.220, 76.228, 76.219, 76.217, 76.215, 76.225, 76.218]

# -------------------------------
# 1a. OpenWeatherMap API Key
# -------------------------------
st.sidebar.subheader("OpenWeatherMap API")
API_KEY = st.sidebar.text_input("Enter your API Key", type="password")

# -------------------------------
# 1b. Get live rainfall for each zone
# -------------------------------
def get_live_rainfall(lat, lon, api_key):
    try:
        url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}&units=metric"
        res = requests.get(url).json()
        rainfall = res.get("rain", {}).get("1h", 0)
        return rainfall
    except:
        return 0

if API_KEY:
    df["LiveRainfall_mm"] = df.apply(lambda row: get_live_rainfall(row["Lat"], row["Lon"], API_KEY), axis=1)
else:
    st.sidebar.warning("No API Key entered. Using default rainfall 20 mm.")
    df["LiveRainfall_mm"] = 20

df["LiveRainfallNorm"] = df["LiveRainfall_mm"] / 400  # normalize 0-1

# -------------------------------
# 2. Elevation normalization
# -------------------------------
df["ElevationNorm"] = 1 - (df["Elevation"] - df["Elevation"].min()) / (df["Elevation"].max() - df["Elevation"].min())

# -------------------------------
# 3. Adjusted FloodLevel with drainage, soil, elevation
# -------------------------------
df["FloodLevel"] = df["LiveRainfallNorm"] * (1 - df["DrainageEfficiency"]) * (1 - df["SoilPermeability"]) * (1 + df["ElevationNorm"])

# -------------------------------
# 4. PanicScore
# -------------------------------
st.sidebar.subheader("Simulate Social Media Panic")
simulate_panic = st.sidebar.checkbox("Use random social media messages", value=True)
if simulate_panic:
    np.random.seed(42)
    df["FearMessages"] = np.random.randint(0, 100, size=len(df))
else:
    df["FearMessages"] = 50

df["PopNorm"] = df["PopulationDensity"] / df["PopulationDensity"].max()
df["FearNorm"] = df["FearMessages"] / df["FearMessages"].max()
df["PanicScore"] = w_pop*df["PopNorm"] + w_fear*df["FearNorm"]

# -------------------------------
# 5. RiskScore & Rank
# -------------------------------
df["RiskScore"] = alpha*df["FloodLevel"] + beta*df["PopNorm"] + gamma*df["PanicScore"]
df["RiskRank"] = df["RiskScore"].rank(ascending=False).astype(int)

# -------------------------------
# 6. Streamlit Layout
# -------------------------------
st.title("🌆 SynapseCity AI – Live Flood Risk Dashboard")
st.markdown("🌆 Live Flood Risk Dashboard – SynapseCity AI")
selected_zone = st.sidebar.selectbox("Select Zone for Explainability", df["Zone"])

# Zone Data Table
st.subheader("Zone Risk Data")
st.dataframe(df[["Zone","LiveRainfall_mm","FloodLevel","DrainageEfficiency","SoilPermeability","Elevation",
                 "PopulationDensity","FearMessages","PanicScore","RiskScore","RiskRank"]])

# Explainability
zone = df[df["Zone"] == selected_zone].iloc[0]
st.subheader(f"Explainability for {selected_zone}")
st.markdown(f"""
**Adjusted Flood Contribution:** {alpha*zone['FloodLevel']:.2f}  
**Population Contribution:** {beta*zone['PopNorm']:.2f}  
**Panic Contribution:** {gamma*zone['PanicScore']:.2f}  

**Total Risk Score:** {zone['RiskScore']:.2f}  
**Priority Rank:** {zone['RiskRank']}
""")

# Bar Chart
st.subheader("City Risk Visualization")
fig, ax = plt.subplots(figsize=(10,6))
norm = matplotlib.colors.Normalize(vmin=df["RiskScore"].min(), vmax=df["RiskScore"].max())
colors = plt.cm.Reds(norm(df["RiskScore"]))
ax.bar(df["Zone"], df["RiskScore"], color=colors)
ax.set_xlabel("City Zones")
ax.set_ylabel("Risk Score")
ax.set_title("Digital Twin Risk Assessment")
plt.xticks(rotation=45)
sm = plt.cm.ScalarMappable(cmap='Reds', norm=norm)
sm.set_array([])
fig.colorbar(sm, ax=ax, label='Risk Intensity')
st.pyplot(fig)

# -------------------------------
# 7. Interactive Map
# -------------------------------
st.subheader("🌍 Digital Twin Map – Risk & Evacuation Routes")
m = folium.Map(location=[10.320, 76.222], zoom_start=14)
colormap = cm.linear.Reds_09.scale(df["RiskScore"].min(), df["RiskScore"].max())
colormap.caption = "Risk Score"

for _, row in df.iterrows():
    folium.CircleMarker(
        location=[row["Lat"], row["Lon"]],
        radius=10,
        color=colormap(row["RiskScore"]),
        fill=True,
        fill_color=colormap(row["RiskScore"]),
        fill_opacity=0.7,
        popup=(f"{row['Zone']}<br>"
               f"Rainfall: {row['LiveRainfall_mm']:.2f} mm<br>"
               f"FloodLevel: {row['FloodLevel']:.2f}<br>"
               f"Drainage: {row['DrainageEfficiency']:.2f}<br>"
               f"SoilPermeability: {row['SoilPermeability']:.2f}<br>"
               f"Elevation: {row['Elevation']} m<br>"
               f"PanicScore: {row['PanicScore']:.2f}<br>"
               f"RiskScore: {row['RiskScore']:.2f}")
    ).add_to(m)

# Evacuation routes
evac_routes_zones = [
    ["Town Center", "Riverbank Area", "Industrial Zone"],
    ["Market Area", "Railway Station", "Industrial Zone"],
    ["Residential South", "Temple Road", "College Road"]
]

route_risks = [df[df["Zone"].isin(route)]["RiskScore"].mean() for route in evac_routes_zones]
safest_index = np.argmin(route_risks)
safest_route = evac_routes_zones[safest_index]
st.markdown(f"**Safest Route:** {' → '.join(safest_route)}  \nAverage RiskScore: {route_risks[safest_index]:.2f}")

for i, route in enumerate(evac_routes_zones):
    coords = df[df["Zone"].isin(route)][["Lat","Lon"]].values.tolist()
    color = "green" if i==safest_index else "blue"
    folium.PolyLine(
        locations=coords, color=color, weight=5, opacity=0.8,
        tooltip="Evacuation Route (Safest)" if i==safest_index else "Evacuation Route"
    ).add_to(m)

colormap.add_to(m)
st_folium(m, width=700, height=500)