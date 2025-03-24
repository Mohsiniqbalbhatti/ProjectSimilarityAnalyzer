**Project Similarity Analyzer - Setup Guide**

---

### **Prerequisites**

Before starting, make sure the following are installed and set up on your system:

- **Node.js** (Required for frontend)
- **Python** (Required for backend)
- **Redis** (Ensure Redis is installed and running)

---

### **Step 1: Extract the ZIP File**

1. Download and extract the provided ZIP file named **"Project Similarity Analyzer.zip"**.
2. Open the extracted folder, which contains two subfolders:
   - **backend** (Backend code)
   - **frontend** (Frontend code)

---

### **Step 2: Running the Backend**

1. Open **VS Code** or **Command Prompt (cmd/PowerShell)**.
2. Navigate to the `backend` directory:
   ```sh
   cd backend
   ```
3. Install required Python packages:
   ```sh
   pip install Flask Flask-SQLAlchemy Flask-Cors numpy sentence-transformers transformers scikit-learn redis werkzeug pyjwt
   ```
4. Start the backend server by running:
   ```sh
   python app.py
   ```
5. The backend will start running on:
   ```
   http://localhost:5000
   ```
6. Open a browser and go to `localhost:5000` to check if the backend is running correctly.
7. **Important:** Make sure **Redis** is installed and running before starting the backend.

---

### **Step 3: Running the Frontend**

1. Open a new **terminal** (Command Prompt, PowerShell, or VS Code terminal).
2. Navigate to the `frontend` directory:
   ```sh
   cd frontend
   ```
3. Install dependencies by running:
   ```sh
   npm install
   ```
   _(This will install all required frontend packages.)_
4. Start the frontend development server:
   ```sh
   npm run dev
   ```
5. The frontend will start running on:
   ```
   http://localhost:5173
   ```
6. Open a browser and go to `localhost:5173` to use the application.

---

### **Additional Notes:**

- Ensure **Python** and **Node.js** are installed on your system.
- If Redis is not installed, follow the official documentation to set it up.
- If you face any issues, check error logs in the terminal and ensure all dependencies are installed.

---

**Your application should now be running successfully!** 🚀
