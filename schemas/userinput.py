from fastapi import FastAPI 
from pydantic import BaseModel, field_validator,computed_field,Field
from typing import Annotated,Literal



class UserInput(BaseModel):
    """
    Pydantic model representing user health and demographic data.
    
    This model validates incoming API requests, ensures physical measurements 
    are mathematically valid, computes derived metrics (like BMI), and formats 
    the data for machine learning model inference.
    """
    
    gender: Annotated[Literal["Male", "Female"], Field(..., description="Gender")]
    region: Annotated[Literal["Urban", "Rural"], Field(..., description="Region")]
    weight: Annotated[float, Field(...,gt=0, description="Weight in kg (wt)")]
    height: Annotated[float, Field(...,gt=0, description="Height in cm (to calculate BMI)")]
    waist: Annotated[float, Field(...,gt=0, description="Waist in inches (wst)")]
    systolic: Annotated[int, Field(...,gt=0, description="Systolic BP (sys)")]
    diastolic: Annotated[int, Field(...,gt=0, description="Diastolic BP (dia)")]
    family_history: Annotated[Literal["Yes", "No"], Field(..., description="Family History (his)")]
    thirst: Annotated[Literal["Yes", "No"], Field(..., description="Excessive Thirst (dipsia)")]
    urination: Annotated[Literal["Yes", "No"], Field(..., description="Frequent Urination (uria)")]
    hdl: Annotated[float, Field(..., description="HDL Cholesterol")]
    exercise_hours: Annotated[float, Field(..., description="Weekly Exercise (Exr_hours)")]

    @computed_field
    @property
    def bmi(self) -> float:
        """
        Calculates Body Mass Index (BMI) automatically from height and weight.
        
        Returns:
            float: The calculated BMI rounded to 2 decimal places.
        """
        return round(self.weight / ((self.height / 100) ** 2), 2)
    

    def to_model_input(self) -> list[float | int]:
        """
        Converts the validated model data into a numerical list for ML prediction.
        
        Categorical variables (Gender, Region, Yes/No fields) are encoded into 
        binary integers (1 or 0).
        
        Returns:
            list[float | int]: A list of numerical features ordered correctly for the ML model.
        """
        m = {"Male": 1, "Female": 0, "Urban": 1, "Rural": 0, "Yes": 1, "No": 0}
        
        return [
            m[self.gender], 
            m[self.region], 
            self.weight, 
            self.bmi,
            self.waist, 
            self.systolic, 
            self.diastolic, 
            m[self.family_history], 
            m[self.thirst], 
            m[self.urination], 
            self.hdl, 
            self.exercise_hours
        ]
