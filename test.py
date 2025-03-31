
from fastapi import FastAPI, HTTPException
import joblib

app = FastAPI()



# Load the bond allocation model (ensure the model path is correct)
bond_model = joblib.load('models/bond_allocation_model.pkl')

@app.post("/bond_allocation")
def bond_allocation():
    try:
        # Fetch bond allocation from the loaded model
        allocations = bond_model.predict([])  # Adjust this based on how your model works
        
        # Assuming the model returns percentages for bonds A, B, and C
        return {
            'Bond A': round(allocations[0] * 100, 2),
            'Bond B': round(allocations[1] * 100, 2),
            'Bond C': round(allocations[2] * 100, 2)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching bond allocation: {str(e)}")
    
    if __name__ == "__main__":
         uvicorn.run(app, host="127.0.0.1", port=8000)