# 🌏 Eco-Friendly India Trip Planner (EcoTrip-AI)

EcoTrip-AI is an advanced, comprehensive travel itinerary planner specifically designed for exploring India sustainably. It generates optimized, multi-day tours with carbon-aware routing, sustainable transport suggestions, intelligent destination clustering, and a highly interactive visual experience. 

## ✨ Key Features
* **Eco-Friendly Routing:** Prioritizes sustainable travel modes (trains, ferries, buses) and calculates estimated carbon footprints for each leg of the journey based on the chosen mode.
* **Smart Itinerary Generation:** Uses KMeans/Agglomerative Clustering along with Nearest Neighbor & 2-Opt TSP (Traveling Salesperson Problem) solvers to group destinations geographically and optimize daily travel routes.
* **Interactive Map UI:** Visualizes your journey dynamically on an interactive Folium map within a beautiful Streamlit interface. 
* **Custom Transport Hubs Extraction:** Automatically recommends the nearest airports and railway stations for your destinations to simplify long-distance travel, seamlessly factoring them into the itinerary path.
* **Personalized Preferences:** Tailors your trip down to the granular details, checking variables such as user budget, travel group type, accessibility needs, safety ratings, altitude tolerance, and preferred traveling season.
* **FastAPI Backend:** Provides a clean REST API (`/generate_itinerary`) for integrating the powerful routing engine into external client applications.
* **Disk Caching for Performance:** Uses local JSON caching for Geoapify routing queries to dramatically boost performance and lower API call volumes.

## 🛠️ Technology Stack
* **Frontend:** Streamlit, Streamlit-Folium, Streamlit-Lottie
* **Backend:** FastAPI, Uvicorn
* **Data Processing & ML:** Pandas, NumPy, Scikit-Learn
* **Geospatial & Routing:** Geoapify API, Haversine Distance, Folium

## 🚀 Getting Started

### Prerequisites
* Python 3.8+
* Geoapify API Key (Required for high-accuracy driving distance / time routing. Without it, the application cleanly falls back to mathematically simulated Haversine distances).

### Installation
1. Clone the repository to your local machine:
   ```bash
   git clone <your-repo-url>
   cd EcoTrip-AI
   ```
2. Install the required Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up your Streamlit secrets to supply the Geoapify API key:
   Create a `.streamlit/secrets.toml` file in the root directory:
   ```toml
   GEOAPIFY_API_KEY = "your_geoapify_api_key_here"
   ```

### Running the Application

**Option 1: Streamlit Web Application (UI)**
Execute the main frontend entrypoint:
```bash
streamlit run app.py
```
*The web interface will automatically open in your default browser at `http://localhost:8501`.*

**Option 2: FastAPI Backend Engine (API)**
If you wish to use the logic headlessly or act as a backend for another frontend:
```bash
uvicorn api:app --reload
```
*The interactive API documentation & playground (Swagger UI) will run at `http://localhost:8000/docs`.*

## 📂 Architecture overview
* **`app.py`:** The Streamlit frontend responsible for state management, parameter input sidebars, custom styling, animations, and the comprehensive presentation of the itinerary loop.
* **`api.py`:** The FastAPI service mapping request validation layers directly onto the generation engine. 
* **`backend.py`:** Engine core containing data cleaning processes, green-scores, spatial clustering algorithms (TSP/k-means), feature calculations (distance/time/cost/carbon), and Geoapify HTTP interactions.
* **`india_tourism_dataset.json`:** The expansive raw dataset containing nuanced geographical, review, transport hub, and categorical attributes for Indian points of interest.
* **`distance_cache.json`:** System-generated disk cache mitigating repetitive and expensive spatial distance queries.

## 🤝 Contributing
Contributions, issue creations, and feature requests are very welcome! Feel free to check the issues page. Let's make travel greener!

## 📜 License
This project is licensed under the [MIT License](LICENSE) - see the LICENSE file for details.
