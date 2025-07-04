import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
from intervention_optimizer import BankInterventionOptimizer

# Page Config
st.set_page_config(
    page_title="Bank Churn Prevention",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load Data and Model
@st.cache_data
def load_data_and_model():
    df = pd.read_csv("data/predictions/data_with_predictions.csv")
    model = joblib.load("models/catboost.pkl")  # Load trained model
    return df, model

df, model = load_data_and_model()

# Sidebar Controls
with st.sidebar:
    st.title("Configuration")
    risk_threshold = st.slider(
        "Minimum Churn Risk",
        0.5, 1.0, 0.7, 0.01
    )
    max_budget = st.number_input(
        "Retention Budget ($)",
        500, 10000, 2000, 100
    )
    if st.button("🔄 Generate Recommendations"):
        st.session_state.force_update = True

# Initialize
if 'force_update' not in st.session_state:
    st.session_state.force_update = True

# Main App
st.title("Customer Retention Optimizer")

if st.session_state.force_update:
    with st.spinner("Optimizing interventions..."):
        # Get predictions
        X = df.drop(columns=['CustomerId', 'Churn'])  # Features only
        df['churn_prob'] = model.predict_proba(X)[:, 1]  # Get probabilities
        
        # Run optimizer
        optimizer = BankInterventionOptimizer(df)
        high_risk = df[df['churn_prob'] >= risk_threshold]['CustomerId'].values
        results = optimizer.optimize(high_risk, max_budget)
        
        st.session_state.update({
            'allocations': results['allocations'],
            'strategy_counts': results['strategy_counts'],
            'metrics': {
                'targeted': len(results['allocations']),
                'roi': results['total_roi'],
                'budget_used': results['budget_used'],
                'high_risk_count': len(high_risk)
            },
            'force_update': False
        })

# Display Results
if 'allocations' in st.session_state:
    # Metrics
    col1, col2, col3 = st.columns(3)
    col1.metric(
        "High-Risk Customers", 
        st.session_state.metrics['high_risk_count'],
        f"{st.session_state.metrics['targeted']} targeted"
    )
    col2.metric(
        "Total ROI", 
        f"${st.session_state.metrics['roi']:,.0f}"
    )
    col3.metric(
        "Budget Used",
        f"${st.session_state.metrics['budget_used']:,.0f}/${max_budget:,}",
        f"{st.session_state.metrics['budget_used']/max_budget:.0%} utilization"
    )

    # Visualizations
    tab1, tab2 = st.tabs(["Strategy Allocation", "Recommendations"])
    
    with tab1:
        # Strategy distribution
        strat_df = pd.DataFrame.from_dict(
            st.session_state.strategy_counts,
            orient='index',
            columns=['Count']
        ).reset_index()
        
        fig = px.pie(
            strat_df,
            names='index',
            values='Count',
            title="Intervention Distribution",
            hole=0.3
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # Recommendations table
        st.dataframe(
            st.session_state.allocations[
                ['CustomerID', 'Strategy', 'ROI', 'Cost', 'Risk']
            ].sort_values('ROI', ascending=False).style.format({
                'ROI': '${:,.0f}',
                'Cost': '${:,.0f}',
                'Risk': '{:.1%}'
            }),
            height=600,
            use_container_width=True
        )

        # Download button
        st.download_button(
            "💾 Export Recommendations",
            data=st.session_state.allocations.to_csv(index=False),
            file_name="retention_recommendations.csv",
            mime="text/csv"
        )

# Footer
st.markdown("---")
st.caption("Bank Retention System v3.0 | ML-Powered")