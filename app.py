import datetime
import pickle
from unittest import result
from fastapi import FastAPI, Depends, HTTPException, Request, status, File, UploadFile
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from flask import jsonify, request
from flask_jwt_extended import jwt_required
from datetime import datetime, timedelta
from jwt import PyJWTError
import pandas as pd
from keras.api.models import Sequential
from keras.api.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt
from sqlalchemy.orm import Session
from typing import Literal, Dict, Any
from sqlalchemy.exc import IntegrityError
from pydantic import BaseModel, EmailStr
from enum import Enum
# from jose import JWTError, jwt
import logging
from pydantic import BaseModel
from typing import List
from fastapi.templating import Jinja2Templates
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from passlib.context import CryptContext
from fastapi.middleware.cors import CORSMiddleware
# from jose import JWTError, jwt
from datetime import timedelta
import logging
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI, Form, UploadFile
from fastapi.responses import HTMLResponse
import numpy as np
import tensorflow as tf
# from tensorflow import keras
from keras.api.models import load_model
from sklearn.preprocessing import MinMaxScaler
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from typing import Any
import shutil
import os
from pydantic import BaseModel
from fastapi import Request
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from typing import List
import httpx
from database import SessionLocal, engine  # Import your database session and models
from models import User, Portfolio  # Import your SQLAlchemy models
# from security import get_password_hash, verify_password, create_access_token  # Import your security functions
from fastapi import APIRouter, HTTPException
from fastapi.staticfiles import StaticFiles
from starlette.responses import FileResponse  # Import FileResponse
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from fastapi import FastAPI, Form, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import joblib
# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

router = APIRouter()
portfolios = []

# CORS setup (if necessary)
origins = [
    "http://localhost", 
    "http://127.0.0.1:8000",  # For local development
]

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # The origin of your frontend
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods: POST, GET, OPTIONS, etc.
    allow_headers=["*"],  # Allow all headers
)


# Simulated Data
simulated_data = [
    {
        "interest_rate": 2.1,
        "inflation_rate": 1.4,
        "gdp_growth": 2.3,
        "risk_tolerance": 1,
        "investment_horizon": 2,
        "client_goal": 0,
        "predicted_return_a": 7.8,
        "predicted_return_b": 5.5,
        "predicted_return_c": 4.2,
        "volatility_a": 10.2,
        "volatility_b": 6.8,
        "volatility_c": 5.1
    },
     {
            "Interest_Rate": 3.0,
            "Inflation_Rate": 2.0,
            "GDP_Growth": 3.0,
            "Risk_Tolerance": 2,
            "Investment_Horizon": 3,
            "Client_Goal": 1,
            "Predicted_Return_A": 9.2,
            "Predicted_Return_B": 7.0,
            "Predicted_Return_C": 5.5,
            "Volatility_A": 12.5,
            "Volatility_B": 8.0,
            "Volatility_C": 6.0
        },
        {
            "Interest_Rate": 1.5,
            "Inflation_Rate": 1.1,
            "GDP_Growth": 1.8,
            "Risk_Tolerance": 0,
            "Investment_Horizon": 1,
            "Client_Goal": 2,
            "Predicted_Return_A": 5.0,
            "Predicted_Return_B": 3.8,
            "Predicted_Return_C": 2.9,
            "Volatility_A": 8.0,
            "Volatility_B": 5.5,
            "Volatility_C": 4.0
        }
    # Add more simulated records here
]

# Define the data model using Pydantic
class MarketCondition(BaseModel):
    interest_rate: float
    inflation_rate: float
    gdp_growth: float
    risk_tolerance: int
    investment_horizon: int
    client_goal: int
    predicted_return_a: float
    predicted_return_b: float
    predicted_return_c: float
    volatility_a: float
    volatility_b: float
    volatility_c: float

# Serve static files from the static folder inside frontend
app.mount("/static", StaticFiles(directory="frontend/static"), name="static")

# Serve the index.html file on the root endpoint
@app.get("/", response_class=HTMLResponse, summary="Serve the index.html UI")
async def serve_ui():
    file_path = os.path.join("frontend", "index.html")
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Index file not found.")
    
    with open(file_path, "r") as file:
        content = file.read()
    
    return HTMLResponse(content=content)

# Serve the login.html file
@app.get("/login", response_class=HTMLResponse, summary="Serve the login page")
async def get_login_page():
    file_path = os.path.join("frontend", "login.html")  # Adjust the path if necessary
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Login page not found.")
    
    with open(file_path, "r") as file:
        content = file.read()
    
    return HTMLResponse(content=content)

# Load the LSTM model with error handling
try:
    lstm_model = tf.keras.models.load_model('models/stock_lstm_model.keras')
    logger.info("LSTM model loaded successfully.")
except Exception as e:
    logger.error(f"Error loading LSTM model: {e}")
    raise HTTPException(status_code=500, detail="Error loading LSTM model.")

# Load your trained bond prediction model
try:
    bond_model = tf.keras.models.load_model('models/bond_allocation_model.keras')
    scaler = MinMaxScaler(feature_range=(0, 1))
except Exception as e:
    raise HTTPException(status_code=500, detail="Error loading bond prediction model.")

bond_model = joblib.load('models/bond_allocation_model.pkl')  # Replace with your actual model path
commodity_model = np.load('models/q_table.npy', allow_pickle=True) # Replace with your actual model path



# Initialize a scaler for stock prices
scaler = MinMaxScaler(feature_range=(0, 1))

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/token")
def allocate_investments(user_data):
    # Example user_data structure
    risk_tolerance = user_data['risk_tolerance']
    investment_goals = user_data['investment_goals']
    investment_horizon = user_data['investment_horizon']
    
    # Default allocations
    allocation = {
        'bonds': 0,
        'commodities': 0,
    }

    # Allocation logic based on risk tolerance
    if risk_tolerance == 'conservative':
        allocation['bonds'] = 70
        allocation['commodities'] = 30
    elif risk_tolerance == 'moderate':
        allocation['bonds'] = 50
        allocation['commodities'] = 50
    elif risk_tolerance == 'aggressive':
        allocation['bonds'] = 30
        allocation['commodities'] = 70

    # Further adjust allocations based on investment goals
    if investment_goals == 'income':
        allocation['bonds'] += 10  # Increase bonds for income
        allocation['commodities'] -= 10  # Decrease commodities
    elif investment_goals == 'growth':
        allocation['bonds'] -= 10  # Decrease bonds for growth
        allocation['commodities'] += 10  # Increase commodities

    # Adjust for investment horizon
    if investment_horizon == 'short-term':
        allocation['bonds'] += 5  # More bonds for stability
        allocation['commodities'] -= 5  # Less commodities
    elif investment_horizon == 'long-term':
        allocation['bonds'] -= 5  # Fewer bonds for growth
        allocation['commodities'] += 5  # More commodities

    return allocation
# Dummy portfolio generator function
def generate_portfolio(income, risk_tolerance, financial_goals, horizon):
    # This function should generate a portfolio based on the user's data
    # Here's an example of a simple logic
    portfolio = {
        "income": income,
        "risk_tolerance": risk_tolerance,
        "financial_goals": financial_goals,
        "horizon": horizon,
        "suggestions": f"Based on your income of {income} and a {risk_tolerance} risk tolerance, we recommend a diversified portfolio."
    }
    return portfolio
class FinancialForm(BaseModel):
    fullName: str
    email: str
    income: int
    financialGoals: str
    horizon: str
    riskTolerance: str

def generate_recommendations(form_data: FinancialForm) -> List[str]:
    recommendations = []
    if form_data.riskTolerance == "low":
        recommendations.append("Invest in Bonds")
    elif form_data.riskTolerance == "medium":
        recommendations.append("Invest in a mix of Stocks and Bonds")
    else:
        recommendations.append("Invest in Stocks and Real Estate")
    return recommendations


templates = Jinja2Templates(directory="templates")
# Pydantic models
class UserCreate(BaseModel):
    username: str
    password: str

class StockPredictionRequest(BaseModel):
    input_data: list[float]  # Ensure input_data is a list of floats

class ReinforcementLearningRequest(BaseModel):
    state: list[float]  # State input for reinforcement learning
# Pydantic model for request
class BondPredictionRequest(BaseModel):
    input_data: List[float]  # Adjust based on the required input features
# Define the request body schema
class BondAllocationRequest(BaseModel):
    bond_a_price: float
    bond_b_price: float
    bond_c_price: float
    interest_rate: float
    inflation_rate: float
    gdp_growth: float
    risk_tolerance: str  # Ensure this is one of the expected categories
    investment_horizon: str  # Ensure this is one of the expected categories
    client_goal: str  # Ensure this is one of the expected categories
    
class Portfolio(BaseModel):
    user_id: int
    assets: List[str]
    total_value: float
    recommendations: List[str]
    
class BondInput(BaseModel):
    input_data: List[float]  # Adjust the type based on your model's expected input
    
    
# Define models
# Define a Portfolio model (optional)
class Portfolio(BaseModel):
    income: int
    financialGoals: str
    horizon: str
    riskTolerance: str
    # Add more fields as needed
    
class HorizonEnum(str):
    short_term = 'short-term'
    medium_term = 'medium-term'
    long_term = 'long-term'

class RiskEnum(str):
    low = 'low'
    medium = 'medium'
    high = 'high'

# Load the trained bond allocation model
bond_model = joblib.load('models/bond_allocation_model.pkl')

class InputData(BaseModel):
    Risk_Tolerance: str
    Investment_Horizon: str
    Client_Goal: str
    Interest_Rate: float
    Inflation_Rate: float
    GDP_Growth: float

# Load your Q-table model
with open('bond_allocation_model.pkl', 'rb') as f:
    Q_table = pickle.load(f)

# Define client input data structure
class InputData(BaseModel):
    Interest_Rate: float
    Inflation_Rate: float
    GDP_Growth: float
    Risk_Tolerance: str
    Investment_Horizon: str
    Client_Goal: str

# Define action space (allocations to 3 bonds, sum = 100%)
actions = [(a, b, 1 - a - b) for a in np.arange(0, 1.1, 0.1) 
           for b in np.arange(0, 1.1, 0.1) if a + b <= 1]





# Load the Q-table
q_table_file_path = 'models/q_table.npy'  # Change to .npy since we saved the Q-table in that format
q_table = np.load(q_table_file_path)

# Define action space
action_space = [
    [0.5, 0.3, 0.2],
    [0.4, 0.4, 0.2],
    [0.3, 0.5, 0.2],
    [0.4, 0.3, 0.3],
    [0.3, 0.3, 0.4]
]

# Define input model
class ClientData(BaseModel):
    Risk_Tolerance: float
    Investment_Horizon: float
    Commodity_1_Price: float
    Commodity_1_Volatility: float
    Commodity_2_Price: float
    Commodity_2_Volatility: float
    Commodity_3_Price: float
    Commodity_3_Volatility: float
    Inflation: float
    Interest_Rates: float

# Q-Learning Agent.......................................................
class QLearningAgent:
    def __init__(self, state_size, action_size, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.q_table = np.zeros((state_size, action_size))  # Initialize Q-table
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate

    def choose_action(self, state):
        # Epsilon-greedy action selection
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, self.q_table.shape[1])  # Explore
        else:
            return np.argmax(self.q_table[state])  # Exploit (choose action with max Q-value)

    def update_q_value(self, state, action, reward, next_state):
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + self.gamma * self.q_table[next_state][best_next_action]
        self.q_table[state][action] += self.alpha * (td_target - self.q_table[state][action])

    def save_model(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.q_table, f)  # Save Q-table to a file

    def load_model(self, filename):
        with open(filename, 'rb') as f:
            self.q_table = pickle.load(f)  # Load Q-table from a file

# Set parameters
state_size = 3  # Assuming simplified state space (literacy level)
action_size = 5  # Number of available modules (actions)

# Create a Q-Learning agent
agent = QLearningAgent(state_size, action_size)

# Simulate rewards for each action (for simplicity, random reward generation)
def get_reward(literacy_level, action):
    reward_matrix = {
        0: [10, 15, 5, 20, 5],  # Beginner: Higher reward for basic lessons
        1: [5, 10, 20, 25, 5],  # Intermediate: Higher reward for advanced lessons
        2: [5, 5, 25, 30, 10]   # Advanced: Highest reward for advanced topics
    }
    return reward_matrix[literacy_level][action]

# Train the agent with simulated data
def train_agent(client_data, agent, episodes=1000):
    for episode in range(episodes):
        for client in client_data:
            state = int(client[0])  # Use literacy level as the state (simplified)
            action = agent.choose_action(state)  # Agent selects an action
            reward = get_reward(state, action)  # Simulate reward based on state and action
            next_state = min(state + 1, 2)  # Assume literacy improves (simplified)
            agent.update_q_value(state, action, reward, next_state)




@app.post("/register", summary="Register a new user")
async def register(user: UserCreate, db: Session = Depends(get_db)):
    hashed_password = get_password_hash(user.password)
    new_user = User(username=user.username, hashed_password=hashed_password)

    try:
        db.add(new_user)
        db.commit()
        db.refresh(new_user)
        return {"msg": "User registered successfully!"}
    except IntegrityError:
        db.rollback()
        raise HTTPException(status_code=400, detail="Username already registered.")
    


# User login endpoint
# Password context for hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Dummy secret key for JWT
SECRET_KEY = "your_secret_key"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Sample function to create an access token (modify based on your actual implementation)
def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt_required.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

# Sample function to verify passwords (implement it based on your password hashing)
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

# Login endpoint
@app.post("/login", summary="Login and get a token")
async def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == form_data.username).first()
    if not user or not verify_password(form_data.password, user.hashed_password):
        logger.warning(f"Incorrect login attempt for user {form_data.username}.")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token = create_access_token(data={"sub": user.username}, expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    logger.info(f"User {user.username} logged in successfully.")
    return {"access_token": access_token, "token_type": "bearer", "message": "Login successful"}

# Get current user info
@app.get("/users/me", summary="Get current user info")
async def read_users_me(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt_required.decode(token, "your_secret_key", algorithms=["HS256"])
        username: str = payload.get("sub")
        if username is None:
            logger.error("Username not found in token payload.")
            raise credentials_exception
    except PyJWTError:
        logger.error("JWT decoding error.")
        raise credentials_exception

    user = db.query(User).filter(User.username == username).first()
    if user is None:
        logger.error(f"User {username} not found.")
        raise credentials_exception

    return {"username": user.username}

# Serve the dashboard.html file
@app.get("/dashboard.html", response_class=HTMLResponse, summary="Serve the dashboard page")
async def get_dashboard_page():
    file_path = os.path.join("frontend", "dashboard.html")  # Path to dashboard.html
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Dashboard page not found.")
    
    with open(file_path, "r") as file:
        content = file.read()
    
    return HTMLResponse(content=content)

@app.get("/dashboard.html/{user_id}")
async def get_portfolio(user_id: int):
    # Fetch the portfolio based on user ID
    for portfolio in portfolios:
        if portfolio.user_id == user_id:
            return {"portfolio": portfolio}
    raise HTTPException(status_code=404, detail="Portfolio not found")



# Stock price prediction endpoint
@app.post("/predict_stock/", summary="Predict stock price")
async def predict_stock_price(request: StockPredictionRequest):
    try:
        # Convert the input data to a NumPy array
        input_data = np.array(request.input_data)

        # Rescale the input data (ensure scaling is consistent with your training data)
        input_data_scaled = scaler.fit_transform(input_data.reshape(-1, 1))

        # Reshape the data for the LSTM model (samples, timesteps, features)
        input_data_scaled = input_data_scaled.reshape(1, len(input_data_scaled), 1)

        # Make a prediction using the LSTM model
        prediction = lstm_model.predict(input_data_scaled)

        # Reverse the scaling of the predicted value
        predicted_value = scaler.inverse_transform(prediction)

        # Log the prediction
        logger.info(f"Prediction made successfully: {predicted_value}")

        return {"predicted_value": predicted_value.tolist()}
    
    except Exception as e:
        logger.error(f"Stock prediction error: {e}")
        raise HTTPException(status_code=500, detail="Error during stock prediction.")

# Load your LSTM model when the application starts
# model = keras.models.load_model('models/stock_lstm_model.keras')

@app.get("/predicted-stocks")
async def get_predicted_stocks(token: str = Depends(oauth2_scheme)):
    # Perform your authentication and authorization checks here
    # If authentication fails, raise an HTTPException

    try:
        # Example input data for prediction, adjust as necessary
        input_data = np.random.rand(1, 10, 1)  # Adjust the shape according to your model
        predictions = model.predict(input_data)

        # Process predictions into a usable format
        predicted_stocks = [{"name": f"Stock {i+1}", "prediction": float(pred)} for i, pred in enumerate(predictions.flatten())]

        return {"stocks": predicted_stocks}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Define the Pydantic model
class InvestmentRequest(BaseModel):
    username: str
    email: EmailStr
    investmentPurpose: Literal["Retirement", "Education", "Wealth Building"]
    riskTolerance: Literal["Low", "Medium", "High"]
    investmentHorizon: Literal["Short-term", "Medium-term", "Long-term"]
    income: float


# Function to create a portfolio based on user input
def create_portfolio(investment_request: InvestmentRequest) -> Dict[str, Any]:
    portfolio = {}

    # Example logic to create a portfolio
    if investment_request.riskTolerance == "Low":
        portfolio["bonds"] = investment_request.income * 0.70  # 70% bonds
        portfolio["stocks"] = investment_request.income * 0.20  # 20% stocks
        portfolio["commodities"] = investment_request.income * 0.10    # 10% commodities
    elif investment_request.riskTolerance == "Medium":
        portfolio["bonds"] = investment_request.income * 0.50  # 50% bonds
        portfolio["stocks"] = investment_request.income * 0.40  # 40% stocks
        portfolio["commodities"] = investment_request.income * 0.10    # 10% commodities
    elif investment_request.riskTolerance == "High":
        portfolio["bonds"] = investment_request.income * 0.20   # 20% bonds
        portfolio["stocks"] = investment_request.income * 0.70   # 70% stocks
        portfolio["commodities"] = investment_request.income * 0.10     # 10% commodities

    # Additional logic based on investment purpose and horizon can be added here
    portfolio["investmentPurpose"] = investment_request.investmentPurpose
    portfolio["investmentHorizon"] = investment_request.investmentHorizon

    return portfolio

# Define the endpoint
@app.post("/investment")
async def create_investment(investment_request: InvestmentRequest):
    # Create the portfolio based on user input
    portfolio = create_portfolio(investment_request)
    
    return {
        "message": "Portfolio created successfully!",
        "portfolio": portfolio
    }
    
@app.get("/user/{user_id}/bonds", summary="Get user's bond predictions")
async def get_user_bond_predictions(user_id: int):
    try:
        # Fetch bond predictions from the database for the specified user
        bond_predictions = fetch_user_bond_predictions(user_id)
        return {"bond_predictions": bond_predictions}
    
    except Exception as e:
        logger.error(f"Error retrieving bond predictions for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving bond predictions.")
    
@app.get("/form")
async def get_form():
    return FileResponse("frontend/form.html")

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    global latest_portfolio
    if latest_portfolio is None:
        return templates.TemplateResponse("error.html", {"request": request, "message": "No portfolio data available."})

    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "portfolio": latest_portfolio
    })


class Portfolio(BaseModel):
    user_id: int
    assets: List[str]
    total_value: float
    recommendations: List[str]

portfolios = []  # Store portfolios in memory (for demonstration)

@app.post("/submit-form")
async def submit_form(form_data: FinancialForm):
    logging.info("Received data: %s", form_data)

    # Create a portfolio based on the submitted form data
    user_id = len(portfolios) + 1  # Assign a new user ID (or fetch from DB)
    assets = ["Stocks", "Bonds", "Real Estate"]  # Example assets based on form data
    total_value = form_data.income * 10  # Example calculation for total value
    recommendations = generate_recommendations(form_data)  # Function to generate investment recommendations

    portfolio = Portfolio(
        user_id=user_id,
        assets=assets,
        total_value=total_value,
        recommendations=recommendations
    )
    portfolios.append(portfolio)

    return {"success": True, "message": "Form submitted successfully", "portfolio": portfolio}
#..........................................................................................................

# Load Training Data and Implement Q-Learning Model
data = pd.read_csv('bond_data.csv')

# Define action space (allocations to 3 bonds, sum = 100%)
actions = [(a, b, 1 - a - b) for a in np.arange(0, 1.1, 0.1) 
           for b in np.arange(0, 1.1, 0.1) if a + b <= 1]

# Hardcoded constant values
interest_rate = 0.05  # Example: 5%
inflation_rate = 0.02  # Example: 2%
gdp_growth = 0.03  # Example: 3%

# Discretize state into a tuple (for Q-table)
def discretize_state(row):
    risk_mapping = {'Low': 0, 'Medium': 1, 'High': 2}
    horizon_mapping = {'Short': 0, 'Medium': 1, 'Long': 2}
    goal_mapping = {'Income': 0, 'Preservation': 1, 'Growth': 2}

    state = (
        risk_mapping[row['Risk Tolerance']],
        horizon_mapping[row['Investment Horizon']],
        goal_mapping[row['Client Goal']],
        int(interest_rate * 10),  # Discretizing interest rates
        int(inflation_rate * 10),  # Discretizing inflation
        int(gdp_growth * 10)  # Discretizing GDP growth
    )
    return state

# Initialize Q-table
Q_table = {}
alpha = 0.1  # Learning rate
gamma = 0.95  # Discount factor
epsilon = 0.1  # Exploration rate
num_days = 500

# Reward function
def calculate_reward(row, action):
    alloc_A, alloc_B, alloc_C = action
    bond_return = (alloc_A * row['Predicted Return A'] +
                   alloc_B * row['Predicted Return B'] +
                   alloc_C * row['Predicted Return C'])
    risk_penalty = -0.01 * (alloc_A * row['Volatility A'] +
                            alloc_B * row['Volatility B'] +
                            alloc_C * row['Volatility C'])
    return bond_return + risk_penalty

# Q-learning algorithm
for day in range(num_days - 1):
    state = discretize_state(data.iloc[day])
    if state not in Q_table:
        Q_table[state] = np.zeros(len(actions))
    
    if np.random.uniform(0, 1) < epsilon:
        # Explore
        action_idx = np.random.choice(len(actions))
    else:
        # Exploit
        action_idx = np.argmax(Q_table[state])
    
    action = actions[action_idx]
    reward = calculate_reward(data.iloc[day], action)
    
    # Get next state
    next_state = discretize_state(data.iloc[day + 1])
    if next_state not in Q_table:
        Q_table[next_state] = np.zeros(len(actions))
    
    # Bellman update
    Q_table[state][action_idx] = (1 - alpha) * Q_table[state][action_idx] + \
                                 alpha * (reward + gamma * np.max(Q_table[next_state]))

# Save the Q-table model
with open('bond_allocation_model.pkl', 'wb') as f:
    pickle.dump(Q_table, f)


# Define the request model
class ClientData(BaseModel):
    risk_tolerance: str
    investment_horizon: str
    client_goal: str

# Find closest state function
def find_closest_state(client_state, Q_table):
    min_distance = float('inf')
    closest_state = None
    
    for state in Q_table.keys():
        # Calculate a simple distance measure between the states
        distance = np.linalg.norm(np.array(client_state) - np.array(state))
        if distance < min_distance:
            min_distance = distance
            closest_state = state
    
    return closest_state

# Endpoint to allocate bonds based on user input
@app.post("/allocate_bonds/")
async def allocate_bonds(client_data: ClientData):
    row = {
        'Risk Tolerance': client_data.risk_tolerance,
        'Investment Horizon': client_data.investment_horizon,
        'Client Goal': client_data.client_goal,
        'Interest Rate': interest_rate,
        'Inflation Rate': inflation_rate,
        'GDP Growth': gdp_growth
    }
    
    test_state = discretize_state(row)

    if test_state in Q_table:
        best_action_idx = np.argmax(Q_table[test_state])
        best_action = actions[best_action_idx]
        best_action_percentage = np.round(np.array(best_action) * 100, 2)  # Convert to percentage
        return {
            "Bond A": f"{best_action_percentage[0]}%",
            "Bond B": f"{best_action_percentage[1]}%",
            "Bond C": f"{best_action_percentage[2]}%"
        }
    else:
        closest_state = find_closest_state(test_state, Q_table)
        if closest_state is not None:
            best_action_idx = np.argmax(Q_table[closest_state])
            best_action = actions[best_action_idx]
            best_action_percentage = np.round(np.array(best_action) * 100, 2)
            return {
                "Bond A (Closest Match Found)": f"{best_action_percentage[0]}%",
                "Bond B (Closest Match Found)": f"{best_action_percentage[1]}%",
                "Bond C (Closest Match Found)": f"{best_action_percentage[2]}%"
            }
        else:
            raise HTTPException(status_code=404, detail="No suitable state or closest match found.")

# Load the saved Q-table model
with open('bond_allocation_model.pkl', 'rb') as f:
    loaded_Q_table = pickle.load(f)
    print("Loaded Q-table from file.")
    
#..................................................................................................................
# Model to handle forecast requests
class ForecastRequest(BaseModel):
    forecast_horizon: int = 120  # Default forecast for 10 years (120 months)

# Simulate realistic data
def simulate_stock_data(n, mu=0.001, sigma=0.02):
    return np.cumprod(1 + np.random.normal(mu, sigma, n)) * 100

def simulate_bond_data(n, mu=0.0005, sigma=0.01):
    return np.cumprod(1 + np.random.normal(mu, sigma, n)) * 100

def simulate_commodity_data(n, amplitude=10, frequency=12, noise_level=2):
    t = np.linspace(0, 4 * np.pi, n)
    return amplitude * np.sin(frequency * t / (2 * np.pi)) + amplitude + np.random.normal(0, noise_level, n)

# Generate simulated data for stocks, bonds, and commodities
n_points = 200
dates = pd.date_range(start='2010-01-01', periods=n_points, freq='M')
stocks = pd.DataFrame({
    'MSFT': simulate_stock_data(n_points),
    'TSLA': simulate_stock_data(n_points, mu=0.002, sigma=0.03),
    'AAPL': simulate_stock_data(n_points, mu=0.0015, sigma=0.025)
}, index=dates)

bonds = pd.DataFrame({
    'Bond1': simulate_bond_data(n_points),
    'Bond2': simulate_bond_data(n_points, mu=0.0003, sigma=0.008),
    'Bond3': simulate_bond_data(n_points, mu=0.0007, sigma=0.015)
}, index=dates)

commodities = pd.DataFrame({
    'Crude_Oil': simulate_commodity_data(n_points),
    'Corn': simulate_commodity_data(n_points, amplitude=8, frequency=10, noise_level=1.5),
    'Wheat': simulate_commodity_data(n_points, amplitude=12, frequency=14, noise_level=2.5)
}, index=dates)

portfolio = pd.concat([stocks, bonds, commodities], axis=1)

allocation = np.array([0.5, 0.3, 0.2, 0.2, 0.4, 0.4, 1, 0, 0])
portfolio_value = portfolio.dot(allocation)

# Normalization
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_portfolio_value = scaler.fit_transform(portfolio_value.values.reshape(-1, 1))

# Prepare data for LSTM
def create_dataset(data, time_steps=60):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:i + time_steps])
        y.append(data[i + time_steps])
    return np.array(X), np.array(y)

time_steps = 60
X_train, y_train = create_dataset(scaled_portfolio_value, time_steps)

# Reshape input to [samples, time steps, features]
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# Build LSTM Model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(units=25))
model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=1)

# Function to forecast future values
def forecast_future(model, data, time_steps, forecast_horizon):
    future = []
    last_sequence = data[-time_steps:]
    
    for _ in range(forecast_horizon):
        prediction = model.predict(last_sequence.reshape(1, time_steps, 1))
        future.append(prediction[0, 0])
        last_sequence = np.append(last_sequence[1:], prediction, axis=0)
    
    return np.array(future)

# FastAPI Endpoint
@app.post("/forecast")
def forecast_portfolio(request: ForecastRequest):
    forecast_horizon = request.forecast_horizon
    forecast_values = forecast_future(model, scaled_portfolio_value, time_steps, forecast_horizon)
    
    # Reverse scaling
    forecast_values = scaler.inverse_transform(forecast_values.reshape(-1, 1))
    
    # Prepare forecast dates
    forecast_dates = pd.date_range(start=portfolio.index[-1] + pd.DateOffset(months=1), periods=forecast_horizon, freq='M')
    
    # Return the forecast as a dictionary (date, forecast value)
    forecast_result = {str(date.date()): float(value) for date, value in zip(forecast_dates, forecast_values.flatten())}
    
    return {"forecast": forecast_result}
#................................................................................................................
# Define request model
class ClientRequest(BaseModel):
    client_id: int
    literacy_level: int  # 0 (Beginner), 1 (Intermediate), 2 (Advanced)
    portfolio_risk: int  # 0 (Conservative), 1 (Balanced), 2 (Aggressive)
    portfolio_allocation_stocks: int
    portfolio_allocation_bonds: int
    investment_goal: int  # 0 (Short-term), 1 (Growth), 2 (Income)
    age_group: int  # 0 (20-30), 1 (30-40), 2 (40-50)
    investment_experience: int
    quiz_score: int

# Load the trained Q-learning model
class QLearningAgent:
    def __init__(self, state_size, action_size):
        self.q_table = np.zeros((state_size, action_size))  # Initialize Q-table

    def load_model(self, filename):
        with open(filename, 'rb') as f:
            self.q_table = pickle.load(f)  # Load Q-table from a file

    def choose_action(self, state):
        return np.argmax(self.q_table[state])  # Exploit (choose action with max Q-value)

# Load the model
agent = QLearningAgent(state_size=3, action_size=5)
agent.load_model('q_learning_model.pkl')  # Load pre-trained Q-learning model

# Tracker class for client progress
class LearningProgressTracker:
    def __init__(self):
        self.client_progress = {}  # Dictionary to store progress for each client

    def initialize_client(self, client_id):
        # Initialize a new client's learning progress
        self.client_progress[client_id] = {
            'quiz_scores': [],
            'course_completion': 0,
            'engagement_level': 0,  # Number of interactions
            'portfolio_adjustments': 0,
            'modules_completed': []  # List to track completed modules
        }

    def update_quiz_score(self, client_id, score):
        if client_id in self.client_progress:
            self.client_progress[client_id]['quiz_scores'].append(score)

    def mark_module_completed(self, client_id, module_name):
        if client_id in self.client_progress:
            self.client_progress[client_id]['course_completion'] += 1
            self.client_progress[client_id]['modules_completed'].append(module_name)

    def update_engagement(self, client_id):
        if client_id in self.client_progress:
            self.client_progress[client_id]['engagement_level'] += 1

    def update_portfolio_adjustments(self, client_id):
        if client_id in self.client_progress:
            self.client_progress[client_id]['portfolio_adjustments'] += 1

    def get_progress(self, client_id):
        return self.client_progress.get(client_id, "Client not found")

# Initialize progress tracker
tracker = LearningProgressTracker()

# Simulate reward matrix based on literacy level
def get_reward(literacy_level, action):
    reward_matrix = {
        0: [10, 15, 5, 20, 5],  # Beginner: Higher reward for basic lessons
        1: [5, 10, 20, 25, 5],  # Intermediate: Higher reward for advanced lessons
        2: [5, 5, 25, 30, 10]   # Advanced: Highest reward for advanced topics
    }
    return reward_matrix[literacy_level][action]

# Define the `/recommend_module` endpoint
@app.post("/recommend_module")
async def recommend_module(request: ClientRequest):
    client_id = request.client_id

    # Initialize client progress if not already done
    if client_id not in tracker.client_progress:
        tracker.initialize_client(client_id)

    # Use literacy level as the state
    state = request.literacy_level
    action = agent.choose_action(state)  # Agent selects an action (module)
    
    # Define available modules
    modules = ["basic_risk_management", "diversification_strategies", "portfolio_rebalancing", 
               "derivatives_intro", "retirement_planning"]
    recommended_module = modules[action]

    # Mark the module as completed in the tracker
    tracker.mark_module_completed(client_id, recommended_module)

    # Return the recommended module and updated progress
    progress = tracker.get_progress(client_id)
    
    return {
        "recommended_module": recommended_module,
        "client_progress": progress
    }

# Define an endpoint to update client quiz score
@app.post("/update_quiz_score/{client_id}")
async def update_quiz_score(client_id: int, score: int):
    if client_id not in tracker.client_progress:
        raise HTTPException(status_code=404, detail="Client not found")
    
    tracker.update_quiz_score(client_id, score)
    return {"message": f"Quiz score for client {client_id} updated successfully."}

# Endpoint to retrieve client progress
@app.get("/client_progress/{client_id}")
async def get_client_progress(client_id: int):
    progress = tracker.get_progress(client_id)
    if progress == "Client not found":
        raise HTTPException(status_code=404, detail=progress)
    
    return progress
#...................................................................................................
# Load the Q-Table (make sure the path is correct)
q_table_file_path = 'models/q_table.npy'  # Adjust the path if necessary
q_table = np.load(q_table_file_path)

# Load client data (this should match your data structure)
def load_single_client_data(csv_path):
    return pd.read_csv(csv_path)

# Define request body structure
class ClientRequest(BaseModel):
    client_id: int

# Define action space globally for use in the endpoint
action_space = [
    [0.5, 0.3, 0.2],
    [0.4, 0.4, 0.2],
    [0.3, 0.5, 0.2],
    [0.4, 0.3, 0.3],
    [0.3, 0.3, 0.4]
]

# Endpoint to allocate funds
@app.post("/allocate")
async def allocate(request: ClientRequest):
    single_client_id = request.client_id
    
    # Check if client ID is valid
    if single_client_id < 0 or single_client_id >= q_table.shape[0]:
        return {"error": "Invalid client ID"}

    # Use the trained model to allocate funds for the single client
    action_idx = np.argmax(q_table[single_client_id])
    final_allocation = action_space[action_idx]

    return {
        "Commodity 1": final_allocation[0],
        "Commodity 2": final_allocation[1],
        "Commodity 3": final_allocation[2]
    }

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)

