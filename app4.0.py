import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from functools import reduce
from collections import defaultdict
from typing import List, Dict, Tuple, Any, Optional
import time
from abc import ABC, abstractmethod
import hashlib
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

# ========== BIG DATA CLASSES ==========

class DataPartition(ABC):
    @abstractmethod
    def partition(self, data: pd.DataFrame, num_partitions: int) -> List[pd.DataFrame]:
        pass

class RangePartitioner(DataPartition):
    def __init__(self, column: str):
        self.column = column
    
    def partition(self, data: pd.DataFrame, num_partitions: int) -> List[pd.DataFrame]:
        sorted_data = data.sort_values(by=self.column)
        return np.array_split(sorted_data, num_partitions)

class HashPartitioner(DataPartition):
    def __init__(self, column: str):
        self.column = column
    
    def partition(self, data: pd.DataFrame, num_partitions: int) -> List[pd.DataFrame]:
        partitions = [[] for _ in range(num_partitions)]
        for idx, row in data.iterrows():
            hash_val = int(hashlib.md5(str(row[self.column]).encode()).hexdigest(), 16)
            partition_idx = hash_val % num_partitions
            partitions[partition_idx].append(row)
        return [pd.DataFrame(p) if p else pd.DataFrame() for p in partitions]

class DistributedDataFrame:
    def __init__(self, df: pd.DataFrame, num_partitions: int = 4):
        self.original_df = df
        self.num_partitions = num_partitions
        self.partitions: List[pd.DataFrame] = np.array_split(df, num_partitions)
        self.lineage: List[str] = ["Original DataFrame"]
        self.execution_time = 0
    
    def map(self, func) -> 'DistributedDataFrame':
        start_time = time.time()
        new_partitions = [func(partition) for partition in self.partitions]
        self.execution_time = time.time() - start_time
        
        result = DistributedDataFrame.__new__(DistributedDataFrame)
        result.original_df = pd.concat(new_partitions, ignore_index=True)
        result.num_partitions = self.num_partitions
        result.partitions = np.array_split(result.original_df, self.num_partitions)
        result.lineage = self.lineage + [f"MAP: {func.__name__}"]
        result.execution_time = self.execution_time
        return result
    
    def filter(self, predicate) -> 'DistributedDataFrame':
        start_time = time.time()
        new_partitions = [partition[predicate(partition)] for partition in self.partitions]
        new_partitions = [p for p in new_partitions if len(p) > 0]
        self.execution_time = time.time() - start_time
        
        if not new_partitions:
            new_partitions = [pd.DataFrame()]
        
        result = DistributedDataFrame.__new__(DistributedDataFrame)
        result.original_df = pd.concat(new_partitions, ignore_index=True) if new_partitions else pd.DataFrame()
        result.num_partitions = self.num_partitions
        result.partitions = np.array_split(result.original_df, self.num_partitions) if len(result.original_df) > 0 else [pd.DataFrame()]
        result.lineage = self.lineage + ["FILTER"]
        result.execution_time = self.execution_time
        return result
    
    def collect(self) -> pd.DataFrame:
        return self.original_df
    
    def get_statistics(self) -> Dict:
        return {
            "num_partitions": self.num_partitions,
            "total_rows": len(self.original_df),
            "execution_time_ms": self.execution_time * 1000,
            "lineage_depth": len(self.lineage),
            "lineage": self.lineage
        }

class AdvancedMapReduce:
    @staticmethod
    def map_phase(records: pd.DataFrame, key_func, value_func) -> List[Tuple]:
        return [(key_func(row), value_func(row)) for _, row in records.iterrows()]
    
    @staticmethod
    def combine_phase(mapped_data: List[Tuple], combiner_func) -> Dict:
        combined = defaultdict(list)
        for key, value in mapped_data:
            combined[key].append(value)
        return {key: combiner_func(values) for key, values in combined.items()}
    
    @staticmethod
    def shuffle_phase(combined_data: Dict) -> Dict:
        return dict(sorted(combined_data.items()))
    
    @staticmethod
    def reduce_phase(shuffled_data: Dict, reducer_func) -> Dict:
        return {key: reducer_func(values) for key, values in shuffled_data.items()}
    
    @classmethod
    def execute_pipeline(cls, df: pd.DataFrame, key_func, value_func, 
                        combiner_func, reducer_func) -> Tuple[Dict, Dict]:
        start_time = time.time()
        
        mapped = cls.map_phase(df, key_func, value_func)
        combined = cls.combine_phase(mapped, combiner_func)
        shuffled = cls.shuffle_phase(combined)
        reduced = cls.reduce_phase(shuffled, reducer_func)
        
        execution_time = time.time() - start_time
        
        metrics = {
            "total_records": len(df),
            "execution_time_ms": execution_time * 1000,
            "map_output_records": len(mapped),
            "unique_keys": len(reduced)
        }
        
        return reduced, metrics

class TimeSeriesAnalyzer:
    @staticmethod
    def moving_average(series: pd.Series, window: int) -> pd.Series:
        return series.rolling(window=window).mean()
    
    @staticmethod
    def exponential_smoothing(series: pd.Series, alpha: float = 0.3) -> pd.Series:
        return series.ewm(alpha=alpha).mean()
    
    @staticmethod
    def detect_anomalies(series: pd.Series, threshold: float = 2.0) -> np.ndarray:
        z_scores = np.abs((series - series.mean()) / series.std())
        return z_scores > threshold
    
    @staticmethod
    def trend_analysis(series: pd.Series) -> Dict:
        x = np.arange(len(series))
        z = np.polyfit(x, series, 1)
        slope = z[0]
        
        return {
            "trend": "Increasing" if slope > 0 else "Decreasing",
            "slope": slope,
            "strength": abs(slope),
            "r_squared": np.corrcoef(x, series)[0, 1] ** 2
        }

# ========== ML ENGINE ==========

class FitnessMLEngine:
    def __init__(self):
        self.mgi_model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.is_trained = False
        
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        categorical_cols = ['gender', 'activity_type', 'intensity', 'health_condition', 'smoking_status']
        for col in categorical_cols:
            if col in df.columns:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    df[f'{col}_encoded'] = self.label_encoders[col].fit_transform(df[col].astype(str))
                else:
                    try:
                        df[f'{col}_encoded'] = self.label_encoders[col].transform(df[col].astype(str))
                    except:
                        df[f'{col}_encoded'] = 0
        
        if 'Muscle_Growth_Index' not in df.columns:
            df['Muscle_Growth_Index'] = (
                df['fitness_level'] * 100 +
                df['daily_steps'] * 0.001 +
                df['duration_minutes'] * 0.05
            )
        
        return df
    
    def train_mgi_predictor(self, df: pd.DataFrame):
        df = self.prepare_features(df)
        
        feature_cols = ['age', 'bmi', 'duration_minutes', 'calories_burned', 
                       'avg_heart_rate', 'hours_sleep', 'stress_level', 'daily_steps',
                       'hydration_level', 'fitness_level', 'gender_encoded', 
                       'activity_type_encoded', 'intensity_encoded']
        
        X = df[feature_cols].fillna(df[feature_cols].median())
        y = df['Muscle_Growth_Index']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        self.mgi_model = RandomForestRegressor(n_estimators=50, random_state=42, max_depth=10)
        self.mgi_model.fit(X_train_scaled, y_train)
        
        y_pred = self.mgi_model.predict(X_test_scaled)
        rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
        r2 = 1 - (np.sum((y_test - y_pred) ** 2) / np.sum((y_test - y_test.mean()) ** 2))
        
        self.is_trained = True
        
        return {
            'rmse': rmse,
            'r2': r2,
            'feature_importance': dict(zip(feature_cols, self.mgi_model.feature_importances_))
        }
    
    def predict_mgi(self, user_data: Dict) -> float:
        if not self.is_trained:
            return 0.0
        
        feature_cols = ['age', 'bmi', 'duration_minutes', 'calories_burned', 
                       'avg_heart_rate', 'hours_sleep', 'stress_level', 'daily_steps',
                       'hydration_level', 'fitness_level', 'gender_encoded', 
                       'activity_type_encoded', 'intensity_encoded']
        
        features = np.array([[user_data.get(col, 0) for col in feature_cols]])
        features_scaled = self.scaler.transform(features)
        
        return self.mgi_model.predict(features_scaled)[0]
    
    def get_feature_importance(self) -> pd.DataFrame:
        if not self.is_trained:
            return pd.DataFrame()
        
        feature_cols = ['Age', 'BMI', 'Duration', 'Calories', 
                       'Heart Rate', 'Sleep', 'Stress', 'Steps',
                       'Hydration', 'Fitness Level', 'Gender', 
                       'Activity', 'Intensity']
        
        importance_df = pd.DataFrame({
            'Feature': feature_cols,
            'Importance': self.mgi_model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        return importance_df

class WorkoutRecommender:
    @staticmethod
    def get_workout_plan(user_profile: Dict, goal: str = "general_fitness") -> Dict:
        age = user_profile.get('age', 30)
        fitness_level = user_profile.get('fitness_level', 0.5)
        health_condition = user_profile.get('health_condition', 'None')
        bmi = user_profile.get('bmi', 22)
        
        plans = {
            'beginner': {
                'weekly_frequency': 3,
                'session_duration': 30,
                'activities': ['Walking', 'Light Yoga', 'Swimming'],
                'intensity': 'Low'
            },
            'intermediate': {
                'weekly_frequency': 4,
                'session_duration': 45,
                'activities': ['Running', 'Weight Training', 'Cycling', 'Dancing'],
                'intensity': 'Medium'
            },
            'advanced': {
                'weekly_frequency': 5,
                'session_duration': 60,
                'activities': ['Running', 'Weight Training', 'HIIT', 'Cycling'],
                'intensity': 'High'
            }
        }
        
        if fitness_level < 0.3:
            category = 'beginner'
        elif fitness_level < 0.7:
            category = 'intermediate'
        else:
            category = 'advanced'
        
        plan = plans[category].copy()
        
        # Health condition adjustments
        if health_condition != 'None':
            if 'Diabetes' in health_condition:
                plan['activities'] = ['Walking', 'Swimming', 'Cycling']
                plan['notes'] = '⚕️ Monitor blood sugar before/after exercise. Stay hydrated.'
            elif 'Hypertension' in health_condition:
                plan['activities'] = ['Walking', 'Yoga', 'Swimming']
                plan['intensity'] = 'Low to Medium'
                plan['notes'] = '⚕️ Avoid high-intensity exercises. Focus on gradual progression.'
            elif 'Asthma' in health_condition:
                plan['activities'] = ['Swimming', 'Walking', 'Yoga']
                plan['notes'] = '⚕️ Keep inhaler nearby. Warm up thoroughly.'
            elif 'Heart' in health_condition:
                plan['activities'] = ['Walking', 'Light Cycling']
                plan['intensity'] = 'Low'
                plan['weekly_frequency'] = 3
                plan['notes'] = '⚠️ Consult doctor before starting. Monitor heart rate closely.'
        
        # Age adjustments
        if age > 60:
            plan['session_duration'] = min(plan['session_duration'], 40)
            plan['intensity'] = 'Low to Medium'
            plan['activities'] = [a for a in plan['activities'] if a not in ['HIIT']]
            plan['notes'] = plan.get('notes', '') + ' 👴 Focus on flexibility and balance exercises.'
        elif age < 25:
            plan['notes'] = plan.get('notes', '') + ' 💪 Great age for building strength and endurance!'
        
        # Goal adjustments
        if goal == 'weight_loss':
            plan['weekly_frequency'] = min(plan['weekly_frequency'] + 1, 6)
            plan['cardio_focus'] = True
            plan['calorie_target'] = 'Burn 300-500 calories per session'
            plan['diet_tip'] = '🍎 Maintain calorie deficit of 300-500 cal/day'
        elif goal == 'muscle_gain':
            plan['activities'] = ['Weight Training', 'Resistance Training']
            plan['protein_intake'] = '🥩 1.6-2.2g protein per kg body weight'
            plan['rest_days'] = '💤 Allow 48hrs recovery between muscle groups'
        elif goal == 'endurance':
            plan['activities'] = ['Running', 'Cycling', 'Swimming']
            plan['session_duration'] = plan['session_duration'] + 15
            plan['progression'] = '📈 Increase duration by 10% weekly'
        elif goal == 'flexibility':
            plan['activities'] = ['Yoga', 'Pilates', 'Stretching']
            plan['notes'] = plan.get('notes', '') + ' 🧘 Hold stretches for 30-60 seconds'
        
        # BMI adjustments
        if bmi < 18.5:
            plan['notes'] = plan.get('notes', '') + ' ⚠️ Focus on strength training and adequate nutrition.'
        elif bmi > 30:
            plan['activities'] = ['Walking', 'Swimming', 'Cycling']
            plan['intensity'] = 'Low to Medium'
            plan['notes'] = plan.get('notes', '') + ' 🎯 Focus on low-impact exercises to protect joints.'
        
        return plan
    
    @staticmethod
    def get_daily_schedule(plan: Dict, day: int) -> List[Dict]:
        activities = plan.get('activities', ['Walking'])
        duration = plan.get('session_duration', 30)
        intensity = plan.get('intensity', 'Medium')
        
        selected_activity = activities[day % len(activities)]
        
        schedule = []
        
        schedule.append({
            'phase': '🔥 Warm-up',
            'duration': 5,
            'activity': 'Light cardio and dynamic stretching',
            'description': 'Prepare your body: jumping jacks, arm circles, leg swings'
        })
        
        schedule.append({
            'phase': '💪 Main Workout',
            'duration': duration - 10,
            'activity': selected_activity,
            'intensity': intensity,
            'description': f'{selected_activity} with {intensity.lower()} intensity. Stay consistent!'
        })
        
        schedule.append({
            'phase': '🧘 Cool-down',
            'duration': 5,
            'activity': 'Static stretching and breathing',
            'description': 'Gradual recovery: quad stretch, hamstring stretch, deep breathing'
        })
        
        return schedule

class HealthInsightsEngine:
    @staticmethod
    def analyze_user_data(df: pd.DataFrame, participant_id: int = None) -> Dict:
        if participant_id:
            user_data = df[df['participant_id'] == participant_id]
        else:
            user_data = df
        
        if len(user_data) == 0:
            return {}
        
        insights = {
            'activity_patterns': {},
            'health_metrics': {},
            'recommendations': [],
            'warnings': [],
            'strengths': []
        }
        
        activity_counts = user_data['activity_type'].value_counts()
        insights['activity_patterns'] = {
            'most_common': activity_counts.index[0] if len(activity_counts) > 0 else 'Unknown',
            'variety_score': len(activity_counts) / 5,
            'avg_duration': user_data['duration_minutes'].mean(),
            'consistency': len(user_data) / 30
        }
        
        insights['health_metrics'] = {
            'avg_sleep': user_data['hours_sleep'].mean(),
            'avg_stress': user_data['stress_level'].mean(),
            'avg_steps': user_data['daily_steps'].mean(),
            'avg_hydration': user_data['hydration_level'].mean(),
            'bmi': user_data['bmi'].iloc[-1] if len(user_data) > 0 else 0,
            'avg_heart_rate': user_data['avg_heart_rate'].mean()
        }
        
        # Recommendations
        if insights['health_metrics']['avg_sleep'] < 7:
            insights['recommendations'].append('😴 **Sleep**: Aim for 7-9 hours. Better sleep = better gains!')
        else:
            insights['strengths'].append('✅ Great sleep habits!')
        
        if insights['health_metrics']['avg_stress'] > 7:
            insights['recommendations'].append('🧘 **Stress**: High stress detected. Try yoga, meditation, or rest days.')
            insights['warnings'].append('⚠️ Chronic stress impacts recovery and health')
        elif insights['health_metrics']['avg_stress'] < 5:
            insights['strengths'].append('✅ Excellent stress management!')
        
        if insights['health_metrics']['avg_steps'] < 5000:
            insights['recommendations'].append('👟 **Steps**: Try to reach 7,000-10,000 daily steps.')
        elif insights['health_metrics']['avg_steps'] > 10000:
            insights['strengths'].append('✅ Crushing your step goals!')
        
        if insights['health_metrics']['avg_hydration'] < 2.0:
            insights['recommendations'].append('💧 **Hydration**: Increase water intake to 2.5-3L daily.')
        else:
            insights['strengths'].append('✅ Well hydrated!')
        
        bmi = insights['health_metrics']['bmi']
        if bmi < 18.5:
            insights['warnings'].append('⚠️ BMI indicates underweight. Consider consulting a healthcare provider.')
        elif 18.5 <= bmi < 25:
            insights['strengths'].append('✅ Healthy BMI range!')
        elif bmi > 30:
            insights['warnings'].append('⚠️ High BMI. Focus on gradual, sustainable weight loss.')
        
        if insights['activity_patterns']['variety_score'] < 0.4:
            insights['recommendations'].append('🎯 **Variety**: Mix up your workouts to prevent plateaus.')
        else:
            insights['strengths'].append('✅ Great workout variety!')
        
        return insights

# ========== APP CONFIGURATION ==========

st.set_page_config(page_title="Smart Fitness Coach", page_icon="💪", layout="wide")

st.markdown("""
    <style>
    div[data-baseweb="tab"] {
        transition: all 0.3s ease-in-out;
        font-weight: 600;
        color: #E0E0E0 !important;
    }
    div[data-baseweb="tab"]:hover {
        color: #00E0FF !important;
        text-shadow: 0 0 15px rgba(0, 224, 255, 0.6);
    }
    [data-testid="stSidebar"] {
        background: rgba(18, 22, 30, 0.7) !important;
        backdrop-filter: blur(18px);
    }
    .stAlert {
        background-color: rgba(0, 224, 255, 0.1) !important;
        border-left: 4px solid #00E0FF !important;
    }
    .metric-card {
        background: rgba(30, 34, 42, 0.6);
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #00E0FF;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h1 style="font-size: 3.5rem; color: #00E0FF; text-shadow: 0 0 25px rgba(0, 224, 255, 0.6);">💪 Smart Fitness Coach AI</h1>
        <p style="color: #A0A0A0; font-size: 1.2rem;">Your Personal AI-Powered Fitness & Health Assistant</p>
    </div>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    df = pd.read_csv("fitlife_dataset.csv")
    df.columns = df.columns.str.strip().str.replace(" ", "_")
    df["Muscle_Growth_Index"] = (
        df["fitness_level"] * 100 +
        df["daily_steps"] * 0.001 +
        df["duration_minutes"] * 0.05
    )
    return df

@st.cache_resource
def train_ml_models():
    df = load_data()
    ml_engine = FitnessMLEngine()
    metrics = ml_engine.train_mgi_predictor(df)
    return ml_engine, metrics

ml_engine, model_metrics = train_ml_models()

# Initialize session state
if 'user_profile' not in st.session_state:
    st.session_state.user_profile = None
if 'workout_history' not in st.session_state:
    st.session_state.workout_history = []

# ========== TABS ==========

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🏠 Dashboard",
    "🎯 MGI Calculator",
    "📋 Workout Planner",
    "📊 Progress Tracker",
    "🧠 ML Insights",
    "🔬 BDA Analytics"
])

# ==================== TAB 1: DASHBOARD ====================
with tab1:
    st.header("🏠 Your Personal Fitness Dashboard")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("👤 Your Profile")
        
        with st.form("user_profile_form"):
            age = st.number_input("Age", min_value=18, max_value=100, value=30)
            gender = st.selectbox("Gender", ["M", "F"])
            height = st.number_input("Height (cm)", min_value=100, max_value=250, value=170)
            weight = st.number_input("Weight (kg)", min_value=30, max_value=200, value=70)
            
            health_condition = st.selectbox("Health Condition", 
                ["None", "Diabetes", "Hypertension", "Asthma", "Heart Disease"])
            smoking = st.selectbox("Smoking Status", ["Never", "Former", "Current"])
            
            hours_sleep = st.slider("Average Sleep (hours)", 4, 12, 7)
            stress_level = st.slider("Stress Level (1-10)", 1, 10, 5)
            daily_steps = st.number_input("Daily Steps", 1000, 30000, 7000)
            hydration = st.slider("Daily Water Intake (L)", 0.5, 5.0, 2.0, 0.1)
            
            # Estimate fitness level
            fitness_estimate = min(1.0, (daily_steps / 10000) * 0.5 + (hours_sleep / 9) * 0.3 + (1 - stress_level / 10) * 0.2)
            
            submit_profile = st.form_submit_button("💾 Save Profile", use_container_width=True)
            
            if submit_profile:
                bmi = weight / ((height/100) ** 2)
                st.session_state.user_profile = {
                    'age': age,
                    'gender': gender,
                    'height_cm': height,
                    'weight_kg': weight,
                    'bmi': bmi,
                    'health_condition': health_condition,
                    'smoking_status': smoking,
                    'hours_sleep': hours_sleep,
                    'stress_level': stress_level,
                    'daily_steps': daily_steps,
                    'hydration_level': hydration,
                    'fitness_level': fitness_estimate,
                    'resting_heart_rate': 70,
                    'blood_pressure_systolic': 120,
                    'blood_pressure_diastolic': 80
                }
                st.success("✅ Profile saved successfully!")
                st.rerun()
    
    with col2:
        st.subheader("📊 Health Overview")
        
        if st.session_state.user_profile:
            profile = st.session_state.user_profile
            
            # BMI Analysis
            bmi = profile['bmi']
            if bmi < 18.5:
                bmi_category = "Underweight"
                bmi_color = "🟡"
            elif bmi < 25:
                bmi_category = "Normal"
                bmi_color = "🟢"
            elif bmi < 30:
                bmi_category = "Overweight"
                bmi_color = "🟡"
            else:
                bmi_category = "Obese"
                bmi_color = "🔴"
            
            metric_col1, metric_col2, metric_col3 = st.columns(3)
            with metric_col1:
                st.metric("BMI", f"{bmi:.1f}", f"{bmi_color} {bmi_category}")
            with metric_col2:
                sleep_status = "😴 Good" if profile['hours_sleep'] >= 7 else "⚠️ Low"
                st.metric("Sleep", f"{profile['hours_sleep']}h", sleep_status)
            with metric_col3:
                step_status = "✅ Great" if profile['daily_steps'] >= 7000 else "📈 Improve"
                st.metric("Daily Steps", f"{profile['daily_steps']:,}", step_status)
            
            st.write("---")
            
            # Health Score
            health_score = 0
            score_breakdown = []
            
            if 18.5 <= bmi < 25:
                health_score += 25
                score_breakdown.append("✅ Healthy BMI (+25)")
            else:
                score_breakdown.append("❌ BMI outside healthy range (0)")
                
            if profile['hours_sleep'] >= 7:
                health_score += 25
                score_breakdown.append("✅ Good sleep (+25)")
            else:
                score_breakdown.append("❌ Insufficient sleep (0)")
                
            if profile['stress_level'] <= 5:
                health_score += 20
                score_breakdown.append("✅ Well-managed stress (+20)")
            else:
                score_breakdown.append("❌ High stress (0)")
                
            if profile['daily_steps'] >= 7000:
                health_score += 20
                score_breakdown.append("✅ Active lifestyle (+20)")
            else:
                score_breakdown.append("❌ Low activity (0)")
                
            if profile['hydration_level'] >= 2:
                health_score += 10
                score_breakdown.append("✅ Well hydrated (+10)")
            else:
                score_breakdown.append("❌ Low hydration (0)")
            
            st.subheader(f"Overall Health Score: {health_score}/100")
            st.progress(health_score / 100)
            
            with st.expander("📋 Score Breakdown"):
                for item in score_breakdown:
                    st.write(item)
            
            if health_score >= 80:
                st.success("🎉 Excellent! You're maintaining great health habits!")
            elif health_score >= 60:
                st.info("👍 Good job! Small improvements can make a big difference.")
            else:
                st.warning("⚠️ Let's work on improving your health metrics together!")
            
            st.write("---")
            
            # Quick Recommendations
            st.subheader("💡 Today's Quick Tips")
            
            df = load_data()
            insights = HealthInsightsEngine.analyze_user_data(df)
            
            if insights.get('strengths'):
                st.success("**Your Strengths:**")
                for strength in insights['strengths'][:2]:
                    st.write(strength)
            
            if insights.get('recommendations'):
                st.info("**Recommendations:**")
                for rec in insights['recommendations'][:3]:
                    st.write(rec)
            
            if insights.get('warnings'):
                for warn in insights['warnings']:
                    st.warning(warn)
        else:
            st.info("👈 Please fill in your profile to see personalized insights!")
            st.markdown("""
            **Why create a profile?**
            - Get personalized workout recommendations
            - Calculate your Muscle Growth Index (MGI)
            - Track your progress over time
            - Receive AI-powered health insights
            """)

# ==================== TAB 2: MGI CALCULATOR ====================
with tab2:
    st.header("🎯 Muscle Growth Index (MGI) Calculator")
    
    st.markdown("""
    <div style='background: rgba(0, 224, 255, 0.1); padding: 1rem; border-radius: 10px; border-left: 4px solid #00E0FF;'>
    <h4>What is MGI?</h4>
    <p>Your <strong>Muscle Growth Index</strong> is an AI-powered metric that predicts your muscle 
    development potential based on your activity, fitness level, and lifestyle factors.</p>
    <p><strong>Score Ranges:</strong></p>
    <ul>
    <li>0-50: Beginner (Building Foundation)</li>
    <li>50-100: Intermediate (Making Progress)</li>
    <li>100-150: Advanced (Strong Performance)</li>
    <li>150+: Elite (Exceptional Fitness)</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.write("")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📝 Activity Details")
        
        if not st.session_state.user_profile:
            st.warning("⚠️ Please create your profile in the Dashboard tab first!")
        else:
            activity_type = st.selectbox("Activity Type", 
                ["Running", "Weight Training", "Cycling", "Swimming", "Dancing", "Yoga", "Walking"])
            duration = st.number_input("Duration (minutes)", 10, 180, 45)
            intensity = st.selectbox("Intensity Level", ["Low", "Medium", "High"])
            calories = st.number_input("Estimated Calories Burned", 50, 1500, 300)
            heart_rate = st.number_input("Average Heart Rate (bpm)", 60, 200, 120)
            
            # Helpful tips
            with st.expander("💡 Help me estimate values"):
                st.markdown("""
                **Calorie Estimates (per 30 min):**
                - Walking: 100-150 cal
                - Running: 300-400 cal
                - Cycling: 200-300 cal
                - Swimming: 250-350 cal
                - Weight Training: 150-250 cal
                
                **Heart Rate Zones:**
                - Low intensity: 100-120 bpm
                - Medium intensity: 120-150 bpm
                - High intensity: 150-180 bpm
                """)
            
            if st.button("🧮 Calculate My MGI", type="primary", use_container_width=True):
                profile = st.session_state.user_profile
                
                # Encode categorical variables
                user_input = profile.copy()
                user_input.update({
                    'duration_minutes': duration,
                    'calories_burned': calories,
                    'avg_heart_rate': heart_rate,
                    'activity_type': activity_type,
                    'intensity': intensity
                })
                
                # Encode for model
                if 'gender' in ml_engine.label_encoders:
                    user_input['gender_encoded'] = ml_engine.label_encoders['gender'].transform([profile['gender']])[0]
                if 'activity_type' in ml_engine.label_encoders:
                    try:
                        user_input['activity_type_encoded'] = ml_engine.label_encoders['activity_type'].transform([activity_type])[0]
                    except:
                        user_input['activity_type_encoded'] = 0
                if 'intensity' in ml_engine.label_encoders:
                    try:
                        user_input['intensity_encoded'] = ml_engine.label_encoders['intensity'].transform([intensity])[0]
                    except:
                        user_input['intensity_encoded'] = 1
                
                # Predict MGI
                predicted_mgi = ml_engine.predict_mgi(user_input)
                st.session_state.last_mgi = predicted_mgi
                st.session_state.last_activity = {
                    'activity': activity_type,
                    'duration': duration,
                    'intensity': intensity,
                    'mgi': predicted_mgi
                }
                
                # Add to history
                st.session_state.workout_history.append({
                    'date': pd.Timestamp.now(),
                    'activity': activity_type,
                    'duration': duration,
                    'intensity': intensity,
                    'calories': calories,
                    'mgi': predicted_mgi
                })
                
                st.success(f"✨ Your Predicted MGI: **{predicted_mgi:.2f}**")
                st.balloons()
    
    with col2:
        st.subheader("📊 MGI Analysis")
        
        if 'last_mgi' in st.session_state:
            mgi_value = st.session_state.last_mgi
            
            # Determine category
            if mgi_value < 50:
                category = "Beginner"
                color = "#FF6B6B"
                emoji = "🌱"
                advice = "Focus on building consistency and foundational strength. Great start!"
            elif mgi_value < 100:
                category = "Intermediate"
                color = "#4ECDC4"
                emoji = "💪"
                advice = "You're making great progress! Keep pushing with progressive overload."
            elif mgi_value < 150:
                category = "Advanced"
                color = "#45B7D1"
                emoji = "🏆"
                advice = "Excellent performance! Optimize your training for peak results."
            else:
                category = "Elite"
                color = "#00E0FF"
                emoji = "⭐"
                advice = "Outstanding! You're in the top tier of fitness!"
            
            st.markdown(f"""
            <div style='background: {color}20; padding: 1.5rem; border-radius: 10px; border-left: 4px solid {color}; text-align: center;'>
            <h2 style='color: {color}; margin: 0;'>{emoji} {category} Level</h2>
            <h1 style='color: {color}; margin: 0.5rem 0;'>{mgi_value:.1f}</h1>
            <p style='margin: 0;'>{advice}</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.write("")
            
            # Visual gauge
            fig, ax = plt.subplots(figsize=(10, 4))
            
            categories = ['Beginner\n(0-50)', 'Intermediate\n(50-100)', 'Advanced\n(100-150)', 'Elite\n(150+)']
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#00E0FF']
            ranges = [50, 50, 50, 50]
            
            bars = ax.barh(categories, ranges, color=[c + '40' for c in colors], edgecolor=colors, linewidth=2)
            
            # Mark user position
            if mgi_value < 50:
                y_pos = 0
                x_pos = mgi_value
            elif mgi_value < 100:
                y_pos = 1
                x_pos = mgi_value
            elif mgi_value < 150:
                y_pos = 2
                x_pos = mgi_value
            else:
                y_pos = 3
                x_pos = min(mgi_value, 200)
            
            ax.scatter([x_pos], [y_pos], s=500, c='red', marker='D', 
                      edgecolors='white', linewidths=3, zorder=5, label='Your Score')
            
            ax.set_xlabel('MGI Score', fontsize=12, color='white')
            ax.set_xlim(0, 200)
            ax.legend(loc='upper right')
            ax.set_facecolor('#0e1117')
            fig.patch.set_facecolor('#0e1117')
            ax.tick_params(colors='white')
            ax.spines['bottom'].set_color('white')
            ax.spines['left'].set_color('white')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            
            st.write("---")
            
            # Improvement tips
            st.subheader("📈 How to Improve Your MGI")
            
            col_a, col_b = st.columns(2)
            
            with col_a:
                st.markdown("**Key Factors:**")
                feature_importance = ml_engine.get_feature_importance()
                
                if not feature_importance.empty:
                    top_features = feature_importance.head(5)
                    for idx, row in top_features.iterrows():
                        importance_pct = row['Importance'] * 100
                        st.write(f"• **{row['Feature']}**: {importance_pct:.1f}%")
            
            with col_b:
                st.markdown("**Action Items:**")
                if mgi_value < 50:
                    st.write("• Start with 3 workouts/week")
                    st.write("• Focus on proper form")
                    st.write("• Increase daily steps to 7K+")
                elif mgi_value < 100:
                    st.write("• Increase workout intensity")
                    st.write("• Add variety to exercises")
                    st.write("• Ensure 7-8 hours sleep")
                else:
                    st.write("• Fine-tune nutrition timing")
                    st.write("• Optimize recovery periods")
                    st.write("• Consider periodization")
        else:
            st.info("👈 Calculate your MGI to see detailed analysis!")

# ==================== TAB 3: WORKOUT PLANNER ====================
with tab3:
    st.header("📋 AI-Powered Workout Planner")
    
    if not st.session_state.user_profile:
        st.warning("⚠️ Please create your profile in the Dashboard tab first!")
    else:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("🎯 Your Fitness Goal")
            
            goal = st.selectbox("Select Your Primary Goal", [
                "general_fitness",
                "weight_loss",
                "muscle_gain",
                "endurance",
                "flexibility",
                "stress_relief"
            ], format_func=lambda x: x.replace('_', ' ').title())
            
            st.write("")
            
            if st.button("🚀 Generate My Plan", type="primary", use_container_width=True):
                st.session_state.current_plan = WorkoutRecommender.get_workout_plan(
                    st.session_state.user_profile, goal
                )
                st.success("✅ Personalized plan generated!")
        
        with col2:
            st.subheader("📅 Your Personalized Plan")
            
            if 'current_plan' in st.session_state:
                plan = st.session_state.current_plan
                
                # Plan overview
                st.markdown(f"""
                <div style='background: rgba(0, 224, 255, 0.1); padding: 1rem; border-radius: 10px;'>
                <h4>Weekly Overview</h4>
                <ul>
                <li><strong>Frequency:</strong> {plan['weekly_frequency']} days/week</li>
                <li><strong>Duration:</strong> {plan['session_duration']} minutes/session</li>
                <li><strong>Intensity:</strong> {plan['intensity']}</li>
                </ul>
                </div>
                """, unsafe_allow_html=True)
                
                st.write("")
                
                # Recommended activities
                st.markdown("**🏃 Recommended Activities:**")
                for i, activity in enumerate(plan['activities'], 1):
                    st.write(f"{i}. {activity}")
                
                # Additional guidance
                if 'notes' in plan:
                    st.info(plan['notes'])
                
                if 'calorie_target' in plan:
                    st.success(f"🔥 {plan['calorie_target']}")
                
                if 'protein_intake' in plan:
                    st.success(plan['protein_intake'])
                
                if 'diet_tip' in plan:
                    st.success(plan['diet_tip'])
                
                st.write("---")
                
                # Daily schedule
                st.subheader("📆 Sample Daily Workout")
                
                day_selector = st.select_slider(
                    "Select Day",
                    options=list(range(1, 8)),
                    value=1,
                    format_func=lambda x: f"Day {x}"
                )
                
                daily_schedule = WorkoutRecommender.get_daily_schedule(plan, day_selector - 1)
                
                for phase in daily_schedule:
                    with st.expander(f"{phase['phase']} - {phase['duration']} min", expanded=True):
                        st.write(f"**Activity:** {phase['activity']}")
                        if 'intensity' in phase:
                            st.write(f"**Intensity:** {phase['intensity']}")
                        st.write(f"**Details:** {phase['description']}")
            else:
                st.info("👈 Click 'Generate My Plan' to see your personalized workout schedule!")

# ==================== TAB 4: PROGRESS TRACKER ====================
with tab4:
    st.header("📊 Progress Tracker")
    
    if not st.session_state.workout_history:
        st.info("🏃 No workouts recorded yet. Use the MGI Calculator to log your first workout!")
    else:
        history_df = pd.DataFrame(st.session_state.workout_history)
        
        st.subheader("📈 Your Workout History")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Workouts", len(history_df))
        with col2:
            st.metric("Total Duration", f"{history_df['duration'].sum():.0f} min")
        with col3:
            st.metric("Total Calories", f"{history_df['calories'].sum():.0f} cal")
        
        st.write("---")
        
        # MGI Progress Chart
        st.subheader("🎯 MGI Progress Over Time")
        
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(range(len(history_df)), history_df['mgi'], 
               marker='o', linewidth=2, markersize=8, color='#00E0FF')
        ax.fill_between(range(len(history_df)), history_df['mgi'], alpha=0.3, color='#00E0FF')
        ax.set_xlabel('Workout Session', fontsize=12, color='white')
        ax.set_ylabel('MGI Score', fontsize=12, color='white')
        ax.set_title('Your MGI Progress', fontsize=14, color='#00E0FF')
        ax.grid(True, alpha=0.3)
        ax.set_facecolor('#0e1117')
        fig.patch.set_facecolor('#0e1117')
        ax.tick_params(colors='white')
        ax.spines['bottom'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
        st.write("---")
        
        col_a, col_b = st.columns(2)
        
        with col_a:
            st.subheader("🏃 Activity Breakdown")
            activity_counts = history_df['activity'].value_counts()
            
            fig, ax = plt.subplots(figsize=(8, 6))
            colors_list = ['#00E0FF', '#4ECDC4', '#45B7D1', '#FF6B6B', '#95E1D3']
            ax.pie(activity_counts.values, labels=activity_counts.index, autopct='%1.1f%%',
                  colors=colors_list[:len(activity_counts)], startangle=90)
            ax.set_title('Activity Distribution', color='white', fontsize=12)
            fig.patch.set_facecolor('#0e1117')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        
        with col_b:
            st.subheader("💪 Intensity Distribution")
            intensity_counts = history_df['intensity'].value_counts()
            
            fig, ax = plt.subplots(figsize=(8, 6))
            intensity_counts.plot(kind='bar', ax=ax, color=['#FF6B6B', '#4ECDC4', '#00E0FF'])
            ax.set_title('Workout Intensity', color='white', fontsize=12)
            ax.set_xlabel('Intensity Level', color='white')
            ax.set_ylabel('Count', color='white')
            ax.set_facecolor('#0e1117')
            fig.patch.set_facecolor('#0e1117')
            ax.tick_params(colors='white')
            plt.xticks(rotation=0)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        
        st.write("---")
        
        # Detailed history table
        st.subheader("📋 Detailed Workout Log")
        display_df = history_df.copy()
        display_df['date'] = display_df['date'].dt.strftime('%Y-%m-%d %H:%M')
        display_df['mgi'] = display_df['mgi'].round(2)
        st.dataframe(display_df, use_container_width=True, hide_index=True)
        
        if st.button("🗑️ Clear History"):
            st.session_state.workout_history = []
            st.rerun()

# ==================== TAB 5: ML INSIGHTS ====================
with tab5:
    st.header("🧠 Machine Learning Insights")
    
    st.markdown("""
    <div style='background: rgba(0, 224, 255, 0.1); padding: 1rem; border-radius: 10px;'>
    <p>Our AI model has been trained on thousands of fitness records to provide you with 
    data-driven insights and predictions. Here's what powers your personalized recommendations!</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.write("")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Model Performance")
        
        st.metric("Model Accuracy (R²)", f"{model_metrics['r2']:.3f}")
        st.metric("Prediction Error (RMSE)", f"{model_metrics['rmse']:.2f}")
        
        if model_metrics['r2'] > 0.8:
            st.success("✅ Excellent model performance!")
        elif model_metrics['r2'] > 0.6:
            st.info("👍 Good model performance")
        
        st.write("")
        st.markdown("**What this means:**")
        st.write("• Higher R² = Better predictions")
        st.write("• Lower RMSE = More accurate")
        st.write("• Model learns from real user data")
    
    with col2:
        st.subheader("📊 Feature Importance")
        
        importance_df = ml_engine.get_feature_importance()
        
        if not importance_df.empty:
            fig, ax = plt.subplots(figsize=(10, 6))
            top_10 = importance_df.head(10)
            ax.barh(top_10['Feature'], top_10['Importance'], color='#00E0FF')
            ax.set_xlabel('Importance Score', color='white')
            ax.set_title('Top 10 Features Affecting MGI', color='#00E0FF', fontsize=14)
            ax.set_facecolor('#0e1117')
            fig.patch.set_facecolor('#0e1117')
            ax.tick_params(colors='white')
            ax.spines['bottom'].set_color('white')
            ax.spines['left'].set_color('white')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
    
    st.write("---")
    
    # Clustering analysis
    st.subheader("👥 User Fitness Profiles")
    
    df = load_data()
    
    # Perform clustering
    cluster_features = ['age', 'bmi', 'fitness_level', 'daily_steps', 'hours_sleep']
    X_cluster = df[cluster_features].fillna(df[cluster_features].median())
    
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(X_cluster)
    
    cluster_names = {
        0: "Casual Exercisers",
        1: "Fitness Enthusiasts", 
        2: "Elite Athletes"
    }
    
    col_a, col_b, col_c = st.columns(3)
    
    for i, (col_viz, cluster_name) in enumerate(zip([col_a, col_b, col_c], cluster_names.values())):
        with col_viz:
            cluster_data = df[df['cluster'] == i]
            st.markdown(f"**{cluster_name}**")
            st.metric("Users", len(cluster_data))
            st.metric("Avg MGI", f"{cluster_data['Muscle_Growth_Index'].mean():.1f}")
            st.metric("Avg Steps", f"{cluster_data['daily_steps'].mean():.0f}")
    
    st.write("")
    st.info("💡 **Insight**: The AI groups users into fitness profiles to provide more targeted recommendations!")

# ==================== TAB 6: BDA ANALYTICS ====================
with tab6:
    st.header("🔬 Big Data Analytics Dashboard")
    
    st.markdown("""
    <div style='background: rgba(0, 224, 255, 0.1); padding: 1rem; border-radius: 10px;'>
    <p>Behind the scenes, we use enterprise-grade Big Data techniques including distributed processing,
    MapReduce, and advanced partitioning strategies to analyze fitness data at scale!</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.write("")
    
    df = load_data()
    
    tab_bda1, tab_bda2, tab_bda3 = st.tabs([
        "⚡ Distributed Processing",
        "📈 Time-Series Analysis", 
        "🎯 Advanced Analytics"
    ])
    
    with tab_bda1:
        st.subheader("⚡ Distributed Data Processing")
        
        if st.button("🚀 Run Distributed Pipeline", type="primary"):
            with st.spinner("Processing data across partitions..."):
                # Create distributed DataFrame
                ddf = DistributedDataFrame(df, num_partitions=8)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Total Partitions", ddf.num_partitions)
                    st.metric("Total Records", f"{len(df):,}")
                    
                    # Map operation
                    def enhance_record(partition):
                        partition = partition.copy()
                        partition["intensity_score"] = partition["fitness_level"] * 0.6
                        return partition
                    
                    ddf_mapped = ddf.map(enhance_record)
                    stats = ddf_mapped.get_statistics()
                    
                    st.metric("Processing Time", f"{stats['execution_time_ms']:.2f} ms")
                
                with col2:
                    st.markdown("**Execution Lineage:**")
                    for i, step in enumerate(stats['lineage'], 1):
                        st.write(f"{i}. {step}")
                
                st.success("✅ Distributed processing completed!")
                
                # MapReduce demo
                st.write("---")
                st.subheader("🔄 MapReduce Pipeline")
                
                amr = AdvancedMapReduce()
                
                def key_func(row):
                    return row["activity_type"]
                
                def value_func(row):
                    return {"mgi": row["Muscle_Growth_Index"]}
                
                def combiner_func(values):
                    mgi_vals = [v["mgi"] for v in values]
                    return {"avg_mgi": sum(mgi_vals) / len(mgi_vals)}
                
                def reducer_func(values):
                    if isinstance(values, dict):
                        return values
                    return {"final_avg": sum(v.get("avg_mgi", 0) for v in values) / len(values)}
                
                reduced_data, metrics = amr.execute_pipeline(
                    df, key_func, value_func, combiner_func, reducer_func
                )
                
                m_col1, m_col2, m_col3 = st.columns(3)
                with m_col1:
                    st.metric("Map Output", f"{metrics['map_output_records']:,}")
                with m_col2:
                    st.metric("Unique Keys", metrics['unique_keys'])
                with m_col3:
                    st.metric("Execution Time", f"{metrics['execution_time_ms']:.2f} ms")
    
    with tab_bda2:
        st.subheader("📈 Time-Series Analysis")
        
        if st.button("🔍 Analyze Trends", type="primary"):
            with st.spinner("Computing time-series metrics..."):
                df_sorted = df.sort_values("duration_minutes").reset_index(drop=True)
                
                ts_analyzer = TimeSeriesAnalyzer()
                mgi_series = df_sorted["Muscle_Growth_Index"]
                
                ma_5 = ts_analyzer.moving_average(mgi_series, 5)
                ma_20 = ts_analyzer.moving_average(mgi_series, 20)
                ema = ts_analyzer.exponential_smoothing(mgi_series)
                anomalies = ts_analyzer.detect_anomalies(mgi_series, threshold=2.0)
                trend_info = ts_analyzer.trend_analysis(mgi_series)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Trend", trend_info["trend"])
                with col2:
                    st.metric("Trend Strength", f"{trend_info['strength']:.4f}")
                with col3:
                    st.metric("Anomalies", int(anomalies.sum()))
                
                st.write("")
                
                # Visualization
                fig, ax = plt.subplots(figsize=(14, 6))
                sample_size = min(1000, len(mgi_series))
                indices = np.linspace(0, len(mgi_series)-1, sample_size, dtype=int)
                
                ax.plot(indices, mgi_series.iloc[indices], label='Original', alpha=0.5, color='#E0E0E0')
                ax.plot(indices, ma_5.iloc[indices], label='MA-5', color='#00E0FF', linewidth=2)
                ax.plot(indices, ma_20.iloc[indices], label='MA-20', color='#0077FF', linewidth=2)
                
                anomaly_indices = np.where(anomalies)[0]
                if len(anomaly_indices) > 0:
                    anomaly_sample = anomaly_indices[anomaly_indices < sample_size]
                    if len(anomaly_sample) > 0:
                        ax.scatter(anomaly_sample, mgi_series.iloc[anomaly_sample], 
                                  color='red', s=100, marker='X', label='Anomalies', zorder=5)
                
                ax.set_xlabel("Sample Index", color='white')
                ax.set_ylabel("Muscle Growth Index", color='white')
                ax.set_title("Time-Series Analysis with Anomaly Detection", color='#00E0FF')
                ax.legend()
                ax.grid(True, alpha=0.3)
                ax.set_facecolor('#0e1117')
                fig.patch.set_facecolor('#0e1117')
                ax.tick_params(colors='white')
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
    
    with tab_bda3:
        st.subheader("🎯 Advanced Analytics")
        
        if st.button("📊 Run Analysis", type="primary"):
            with st.spinner("Computing advanced metrics..."):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Activity Performance**")
                    activity_stats = df.groupby("activity_type").agg({
                        "Muscle_Growth_Index": ["mean", "std", "count"]
                    }).round(2)
                    activity_stats.columns = ["Avg MGI", "Std Dev", "Count"]
                    activity_stats = activity_stats.sort_values("Avg MGI", ascending=False)
                    st.dataframe(activity_stats, use_container_width=True)
                
                with col2:
                    st.markdown("**Intensity Impact**")
                    intensity_stats = df.groupby("intensity").agg({
                        "Muscle_Growth_Index": "mean",
                        "calories_burned": "mean"
                    }).round(2)
                    intensity_stats.columns = ["Avg MGI", "Avg Calories"]
                    st.dataframe(intensity_stats, use_container_width=True)
                
                st.write("---")
                
                # Correlation heatmap
                st.markdown("**Feature Correlations**")
                
                corr_features = ["Muscle_Growth_Index", "fitness_level", "daily_steps", 
                               "duration_minutes", "calories_burned", "hours_sleep"]
                corr_matrix = df[corr_features].corr()
                
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", 
                           center=0, ax=ax, cbar_kws={"label": "Correlation"},
                           square=True, linewidths=1)
                ax.set_title("Feature Correlation Matrix", color='white', fontsize=14, pad=20)
                fig.patch.set_facecolor('#0e1117')
                ax.set_facecolor('#0e1117')
                ax.tick_params(colors='white')
                plt.xticks(rotation=45, ha='right')
                plt.yticks(rotation=0)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
                
                st.write("---")
                
                # Performance metrics
                st.subheader("⚡ System Performance")
                
                perf_col1, perf_col2, perf_col3 = st.columns(3)
                
                with perf_col1:
                    start = time.time()
                    _ = df.groupby("activity_type")["Muscle_Growth_Index"].mean()
                    time1 = (time.time() - start) * 1000
                    st.metric("GroupBy Query", f"{time1:.2f} ms")
                
                with perf_col2:
                    start = time.time()
                    _ = df[df["fitness_level"] > df["fitness_level"].median()]
                    time2 = (time.time() - start) * 1000
                    st.metric("Filter Query", f"{time2:.2f} ms")
                
                with perf_col3:
                    start = time.time()
                    _ = df.groupby(["gender", "intensity"]).agg({
                        "Muscle_Growth_Index": ["mean", "std"]
                    })
                    time3 = (time.time() - start) * 1000
                    st.metric("Complex Agg", f"{time3:.2f} ms")
                
                st.success("✅ All analytics completed successfully!")

# ==================== SIDEBAR ====================
with st.sidebar:
    st.markdown("### 🎮 Quick Actions")
    
    if st.session_state.user_profile:
        st.success("✅ Profile Complete")
        profile = st.session_state.user_profile
        st.write(f"👤 {profile['gender']}, {profile['age']} years")
        st.write(f"📊 BMI: {profile['bmi']:.1f}")
    else:
        st.warning("⚠️ Profile Incomplete")
        st.write("Complete your profile to unlock all features!")
    
    st.write("---")
    
    st.markdown("### 📊 Quick Stats")
    if st.session_state.workout_history:
        st.metric("Workouts Logged", len(st.session_state.workout_history))
        total_duration = sum([w['duration'] for w in st.session_state.workout_history])
        st.metric("Total Time", f"{total_duration} min")
    else:
        st.info("No workouts logged yet")
    
    st.write("---")
    
    st.markdown("### 💡 Daily Tip")
    tips = [
        "💧 Drink water before, during, and after workouts!",
        "😴 Quality sleep is crucial for muscle recovery.",
        "🥗 Protein within 30 min post-workout aids recovery.",
        "🏃 Consistency beats intensity for long-term results.",
        "🧘 Don't skip warm-ups - they prevent injuries!",
        "📱 Track your progress to stay motivated.",
        "🎯 Set SMART goals: Specific, Measurable, Achievable.",
        "💪 Rest days are as important as workout days.",
        "🥤 Consider BCAAs for enhanced recovery.",
        "📈 Progressive overload is key to muscle growth."
    ]
    
    import random
    st.info(random.choice(tips))
    
    st.write("---")
    
    st.markdown("### 🔬 Tech Stack")
    with st.expander("View Details"):
        st.markdown("""
        **ML Models:**
        - Random Forest Regressor
        - K-Means Clustering
        - Gradient Boosting (planned)
        
        **BDA Techniques:**
        - Distributed DataFrames
        - MapReduce Pipeline
        - Hash/Range Partitioning
        - Time-Series Analysis
        
        **Data Processing:**
        - Pandas for analysis
        - Scikit-learn for ML
        - Matplotlib & Seaborn for viz
        """)
    
    st.write("---")
    
    st.markdown("""
    <div style='text-align: center; font-size: 0.8rem; color: #888;'>
    <p>💪 Smart Fitness Coach v1.0</p>
    <p>Powered by AI & Big Data</p>
    </div>
    """, unsafe_allow_html=True)

# ==================== FOOTER ====================
st.write("---")
st.markdown("""
<div style='text-align: center; color: #00E0FF;'>
    <p>🚀 <strong>Smart Fitness Coach AI</strong> | Combining Machine Learning with Enterprise Big Data Analytics</p>
    <p style='font-size: 0.9rem; color: #A0A0A0;'>
    Features: MGI Prediction • Personalized Workouts • Progress Tracking • Health Insights • Distributed Processing
    </p>
</div>
""", unsafe_allow_html=True)

# ==================== WELCOME MESSAGE ====================
if not st.session_state.user_profile and 'welcome_shown' not in st.session_state:
    st.balloons()
    st.toast("👋 Welcome to Smart Fitness Coach AI!", icon="💪")
    st.session_state.welcome_shown = True