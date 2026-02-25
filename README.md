# 🌏 Eco‑Friendly India Trip Planner

An AI‑powered itinerary generator that creates personalized, multi‑day eco‑friendly tours across Indian states. The system intelligently clusters destinations, optimizes daily routes, estimates carbon footprint and budget, and suggests nearest transport hubs – all in a user‑friendly Streamlit web app.

---

## ✨ Features

- **State‑wise or All‑India exploration** – Select specific states or cover the whole country.
- **Smart day allocation** – Automatically adjusts the number of days to visit all selected places (enforces a minimum of 365 days for complete India tours).
- **Geographic clustering** – Uses K‑Means to group nearby destinations into daily clusters.
- **Optimal intra‑day routing** – Applies a nearest‑neighbour + 2‑opt TSP solver to minimise travel distance each day.
- **Feasibility enforcement** – Splits any day exceeding 350 km or 12 hours into smaller, realistic days.
- **Transport hub suggestions** – For every start and end point, displays the nearest airport and railway station (taken directly from the dataset).
- **Carbon footprint & budget estimation** – Calculates daily CO₂ emissions and total trip cost (transport + ₹2000/day accommodation).
- **Travel‑only day option** – If your start location isn’t a tourist spot, day 1 can be marked as just travel.
- **Duplicate‑free destinations** – Aggressive name cleaning ensures each unique place appears only once.
- **Interactive web interface** – Built with Streamlit, featuring dropdowns, sliders, and expandable day‑by‑day itineraries.

---

## 🛠️ Tech Stack

| Component       | Technology                         |
|-----------------|------------------------------------|
| Backend         | Python 3.9+, pandas, numpy, scikit‑learn |
| Routing API     | Geoapify (falls back to Haversine) |
| Frontend        | Streamlit                          |
| Clustering      | K‑Means, silhouette score          |
| Optimisation    | Nearest‑neighbour + 2‑opt TSP      |

---

## 📦 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/eco-india-trip-planner.git
   cd eco-india-trip-planner
