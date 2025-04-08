# 🦠 COVID-19 Variants Detection Analyzer

![Python](https://img.shields.io/badge/Python-3.8%2B-blue) ![Streamlit](https://img.shields.io/badge/Streamlit-1.0%2B-red) ![License](https://img.shields.io/badge/License-MIT-green) ![GitHub Issues](https://img.shields.io/github/issues/yourusername/covid19-detection-analyzer) ![GitHub Stars](https://img.shields.io/github/stars/yourusername/covid19-detection-analyzer?style=social)

A powerful **Streamlit-based web application** designed to analyze SARS-CoV-2 genomic data. Upload your dataset (CSV or Excel) and explore features like data summaries, visualizations, exploratory data analysis (EDA), and Random Forest model training to predict viral variants (e.g., Pangolin lineages).

------------------------------------------------------------------------------------------------------------------------------------------

 🛠️ ***Website link*** - https://vaibhavtripathi.streamlit.app


------------------------------------------------------------------------------------------------------------------------------------------

## 🚀 Features

- **📊 Basic Information**: View dataset summary, top/bottom rows, data types, and column names.
- **🛠️ Data Manipulation**: Identify/remove missing values and perform group-by operations.
- **📈 Data Visualization**: Create interactive charts (Bar, Line, Pie, Scatter, Sunburst, Heatmap, 3D Scatter) using Plotly.
- **🔍 EDA**: Check collinearity and outliers with visual insights.
- **🤖 Model Training**: Train a Random Forest model to predict SARS-CoV-2 variants with accuracy metrics.
- **🎨 Theme Toggle**: Switch between vibrant light and dark themes.
- **📤 File Upload**: Supports CSV and Excel files for flexible data input.

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## 📸 Screenshots

| Feature                | Screenshot                          |
|------------------------|-------------------------------------|
| **Home Page**          | ![Home](![Screenshot 2025-04-07 223054](https://github.com/user-attachments/assets/1261823e-3ec4-4729-a6cb-14e363500ded)|
|
| **Data Visualization** | ![Viz](![Screenshot 2025-04-07 223342](https://github.com/user-attachments/assets/8f96ed83-07a3-40fc-94ac-3b85c723cd4c) |
|
| **Model Training**     | ![Model](![Screenshot 2025-04-07 223515](https://github.com/user-attachments/assets/db1c8310-4f04-4594-a939-246d8b8bb712)|


------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## 🛠️ Installation

Follow these steps to set up the project locally:

### Prerequisites
- Python 3.8 or higher
- Git

### Steps
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/covid19-detection-analyzer.git
   cd covid19-detection-analyzer

2. **Create a Virtual Environment (optional but recommended)**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate

      
3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt


4. **Run the App**:
    ```bash
    streamlit run 1.py

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

📖 Usage
1. **Launch the App**:
   ```bash
   Run streamlit run 1.py and visit http://localhost:8501.
   
2. **Upload Data**:
   ```bash
    Upload a CSV or Excel file containing SARS-CoV-2 genomic data (e.g., with columns like Pangolin, Accession, Length, Collection_Date).
   
4. **Navigate Sections**:
   ```bash
    Use the sidebar to switch between:
  
5. **Basic Information**:
   ```bash
    Explore dataset details.
   
6. **Data Manipulation**:
   ```bash
    Clean and group data.

7. **Data Visualization**:
   ```bash
    Generate charts (run a group-by first).

5. **EDA**:
   ```bash
    Analyze collinearity and outliers.
   
6. **Model Training**:
   ```bash
   Train a Random Forest model on Pangolin variants.
   
7. **Settings**:
   ```bash
   Toggle light/dark theme.
   
***Example Workflow***:
Upload a dataset → Go to "Data Manipulation" → Group by Country and count → Visualize as a Pie chart → Train a model to predict Pangolin.

-----------------------------------------------------------------------------------------------------------------------------------------

🤝 ***Contributing***
Contributions are welcome!


1. **Fork the Repo**:
     ----->Click "Fork" on GitHub.

   
2. **Clone Your Fork**:
      ```bash
       git clone https://github.com/Vaibhav-0111/covid19-detection-analyzer.git

------------------------------------------------------------------------------------------------------------------------------------------

📜 ***License***
This project is licensed under the MIT License.

------------------------------------------------------------------------------------------------------------------------------------------

📧 **Contact**
For questions or suggestions:

***GitHub***: Vaibhav-0111

***Email***: vaibhavtripathi724@gamil.com
      
