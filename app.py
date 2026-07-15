# Credit Card Fraud Detection & Transaction Risk Scoring
# Streamlit Dashboard for Risk Scorer & Fraud Forecasting
# Author: Corbin Couger

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pickle
import os
import json
from statsmodels.tsa.arima.model import ARIMA
import shap

# ===========================================
# === Page Config

st.set_page_config(page_title= 'Fraud Risk Dashboard', page_icon = '💳', layout = 'wide')

# ===========================================
# === Colors for Visualizations

FRAUD_COLOR = 'red'
LEGIT_COLOR = 'green'
NEUTRAL_COLOR = 'blue'

# ===========================================
# === Loading Models, Data, Scaler, etc.
@st.cache_resource
def load_artifacts():
    import tensorflow as tf
    nn_model = tf.keras.models.load_model('models/fraud_nn_model.keras')

    with open('models/xgb_model.pkl', 'rb') as f:
        xgb_model = pickle.load(f)

    with open('models/model_metadata.json', 'r') as f:
        metadata = json.load(f)

    with open('models/arima_model.pkl', 'rb') as f:
        arima_model = pickle.load(f)
    
    hourly = pd.read_csv('data/processed/hourly_fraud_series.csv')

    with open('data/processed/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    return nn_model, xgb_model, metadata, arima_model, hourly, scaler

nn_model, xgb_model, metadata, arima_model, hourly, scaler = load_artifacts()
# thresholds:
NN_THRESHOLD = metadata['nn_threshold']
FEATURE_NAMES = metadata['feature_names']

# ===========================================
# === Sidebar
with st.sidebar:
    st.title('Fraud Risk Dashboard')
    st.caption('Credit Card Fraud Detection & Transaction Risk Scoring')
    st.divider()

    st.subheader('Model Performance')

    nn_metrics = metadata['val_metrics']['nn']
    xgb_metrics = metadata['val_metrics']['xgb']

    st.caption('Keras Neural Network (Deployed)')
    col1, col2 = st.columns(2)
    col1.metric('Precision', f"{nn_metrics['precision']:.4f}")
    col2.metric('Recall', f"{nn_metrics['recall']:.4f}")
    col1.metric('F1 Score' , f"{nn_metrics['f1']:.4f}")
    col2.metric('PR-AUC', f"{nn_metrics['auc_pr']:.4f}")

    st.divider()

    st.caption('XGBoost (SHAP Explainability)')
    col3, col4 = st.columns(2)
    col3.metric('Precision', f"{xgb_metrics['precision']:.4f}")
    col4.metric('Recall', f"{xgb_metrics['recall']:.4f}")
    col3.metric('F1 Score' , f"{xgb_metrics['f1']:.4f}")
    col4.metric('PR-AUC', f"{xgb_metrics['auc_pr']:.4f}")

    st.divider()

    st.caption('ARIMA Forecasting')
    col5, col6 = st.columns(2)
    col5.metric('MAE', f"{metadata.get('forecast_mae', 0):.4f}")
    col6.metric('RMSE', f"{metadata.get('forecast_rmse')}")

    st.divider()

    st.caption('Dataset: Kaggle Credit Card Fraud')
    st.caption('Transactions: 284,807 | Fraud Rate: .0172%')
    st.caption('Author: Corbin Couger')

# ===========================================
# === Main Header

st.title('💳 Credit Card Fraud Detection & Risk Scoring')
st.write('End-to-End ML system combining real-time transaction risk scoring' 'with fraud volume forecasting. Built with XGBoost, Keras, SHAP, and ARIMA.')
st.divider()

# ===========================================
# === Tabs
tab1, tab2 = st.tabs(['Transaction Risk Scorer', 'Fraud Forecasting'])

# ===========================================
# === Tab 1, Transaction Risk Scorer
with tab1:
    st.subheader('Transaction Risk Scorer')
    st.write('Enter transaction features below. The model returns a continuous risk' 'score (0 - 1) with SHAP explanation of the top contributing facotrs.')

    # input:
    input_mode = st.radio('Input mode', ['Manual entry', 'Random legitimate transaction', 'Random fraud transaction'], horizontal=True)
    @st.cache_data
    def load_raw():
        return pd.read_csv('data/raw/creditcard.csv').drop_duplicates()
    def load_processed():
        X = pd.read_csv('data/processed/X_val.csv')
        y = pd.read_csv('data/processed/y_val.csv').squeeze()
        return X, y

    X_val_df, y_val_df = load_processed()

    if input_mode == 'Random legitimate transaction':
        sample = X_val_df[y_val_df == 0].sample(1, random_state=np.random.randint(0,9999))
        input_values = sample[FEATURE_NAMES].values[0]
        for feat, val in zip(FEATURE_NAMES, input_values):
            st.session_state[f'input_{feat}'] = float(val)
        st.info('Loaded random legitimate transaction from dataset')
    elif input_mode == 'Random fraud transaction':
        fraud_df = X_val_df[y_val_df == 1]
        fraud_array = fraud_df[FEATURE_NAMES].values.astype('float32')
        fraud_scores = nn_model.predict(fraud_array, verbose=0).flatten()
        caught_mask = fraud_scores >= NN_THRESHOLD
        caught_df = fraud_df[caught_mask]
        sample = caught_df.sample(1, random_state=np.random.randint(0, 9999))
        input_values = sample[FEATURE_NAMES].values[0]
        for feat, val in zip(FEATURE_NAMES, input_values): # ensuring the app doesn't cache any values so this displays an actual fraudulent transaction
            st.session_state[f'input_{feat}'] = float(val)
        st.warning('Loaded a known fraudulent transaction from dataset')
    else:
        input_values = [0.0] * len(FEATURE_NAMES)

    # feature input:
    with st.expander('Edit transaction features', expanded=(input_mode == 'Manual Entry')):
        st.caption('V1-V28 are PCA-transformed features. Amount and Time are raw values.')
        cols = st.columns(5)
        user_inputs = {}
        for i, feat in enumerate(FEATURE_NAMES):
            col = cols[i % 5]
            user_inputs[feat] = col.number_input(feat, value = float(input_values[i]), format = '%.4f', step = .01, min_value = None, max_value = None, key = f'input_{feat}')
        
    # score button:
    score_btn = st.button('Score Transaction', type = 'primary')

    if score_btn:
        input_df = pd.DataFrame([user_inputs])

        if 'Amount_scaled' in input_df.columns:
            input_df['Amount_scaled'] = np.log1p(input_df['Amount_scaled'].abs())

        v_features        = [f'V{i}' for i in range(1, 29)]
        features_to_scale = v_features + ['Amount_scaled', 'Hour_sin', 'Hour_cos']
        scale_cols        = [c for c in features_to_scale if c in input_df.columns]

        if scale_cols:
            input_df[scale_cols] = scaler.transform(input_df[scale_cols])

        input_df    = input_df[FEATURE_NAMES]
        input_array = input_df.values.astype('float32')

        ## score:
        risk_score = float(nn_model.predict(input_array, verbose = 40).flatten()[0])

        # risk tiers:
        if risk_score >= NN_THRESHOLD:
            tier = '🚨 High Risk -> Flagged for Review'
            decision = 'FLAGGED'
        elif risk_score >=.5:
            tier = '⚠️ Medium Risk -> Monitor'
            decision = 'MONITOR'
        else:
            tier = '✅ Low Risk -> Approved'
            decision = 'APPROVED'

        # result:
        if decision == 'FLAGGED':
            st.error(f'{tier} | Risk Score: {risk_score:.4f} | Threshold: {NN_THRESHOLD:4f}')
        elif decision == 'MONITOR':
            st.warning(f'{tier} | Risk Score: {risk_score:.4f} | Threshold: {NN_THRESHOLD:4f}')
        else:
            st.success(f'{tier} | Risk Score: {risk_score:.4f} | Threshold: {NN_THRESHOLD:4f}')

        # metrics:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric('Risk Score', f'{risk_score:.4f}')
        m2.metric('Threshold', f'{NN_THRESHOLD:.4f}')
        m3.metric('Decision', decision)
        m4.metric('Risk Tier', tier.split('-')[0].strip())

        # SHAP:
        st.divider()
        st.subheader('Why Was This Decision Made? (SHAP)')
        st.write('SHAP values show how each feature contributed to this prediction.' 'Red bars push toward fraud while blue bars push toward legitimate')
        with st.spinner('Computing SHAP explanation...'):
            explainer = shap.TreeExplainer(xgb_model)
            shap_vals = explainer.shap_values(input_df[FEATURE_NAMES])
            xgb_score = float(xgb_model.predict_proba(input_df[FEATURE_NAMES])[:,1][0])
            shap_explanation = shap.Explanation(
                values = shap_vals[0],
                base_values = explainer.expected_value,
                data = input_df[FEATURE_NAMES].iloc[0].values,
                feature_names= FEATURE_NAMES
            )
            fig, ax = plt.subplots(figsize = (10, 6))
            shap.plots.waterfall(shap_explanation, max_display=12, show = False)
            plt.title(f'SHAP Waterfall - XGBoost Risk Score: {xgb_score:.4f}')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        
        # feature contribution:
        shap_series = pd.Series(
            np.abs(shap_vals[0]),
            index = FEATURE_NAMES
        ).sort_values(ascending = False).head(10) # top 10 features

        shap_df = pd.DataFrame({
            'Feature': shap_series.index,
            'SHAP (Impact)': shap_series.values.round(4),
            'Feature Value': [round(float(input_df[FEATURE_NAMES].iloc[0][f]), 4) for f in shap_series.index],
            'Direction': ['Toward Fraud' if shap_vals[0][FEATURE_NAMES.index(f)] > 0 else 'Toward Legit' for f in shap_series.index]
        })
        st.subheader('Top 10 Feature Contributinos')
        st.dataframe(shap_df, use_container_width=True, hide_index=True)


# ===========================================
# === Tab 2, Fraud Forecasting

with tab2:
    st.subheader('Fraud Volume Forecasting')
    st.write(
        'ARIMA model forecasting fraud transaction volume. '
        'Proactive spike detection allows fraud teams to tighten detection thresholds '
        'before coordinated fraud attacks.'
    )

    # controls
    horizon = st.slider('Forecast horizon (hours)', min_value = 6, max_value = 24, value = 12, step = 6)

    if st.button('Run Forecast', type = 'primary'):
        with st.spinner('Generating forecast...'):
            fraud_series = hourly['fraud_count'].values
            order_str = metadata.get('forecasting_order', '(0,0,2)')
            p, d, q = [int(x) for x in order_str.strip('()').split((','))]

            arima_refit = ARIMA(fraud_series, order=(p, d, q)).fit()
            forecast_out = arima_refit.get_forecast(steps=horizon)
            forecast_mean = np.clip(forecast_out.predicted_mean, 0, None)
            forecast_ci = forecast_out.conf_int()

            if hasattr(forecast_ci, 'iloc'):
                lower = np.clip(forecast_ci.iloc[:, 0].values, 0, None)
                upper = forecast_ci.iloc[:, 1].values
            else:
                lower = np.clip(forecast_ci[:, 0], 0, None)
                upper = forecast_ci[:,1]

            future_hours = np.arange(hourly['Hour'].max() + 1, hourly['Hour'].max() + 1 + horizon)

        ## forecast metrics
        f1, f2, f3, f4 = st.columns(4)
        f1.metric('Forecast Horizon', f'{horizon} hours')
        f2.metric('Peak Predicted', f'{forecast_mean.max():.1f} fraud/hour')
        f3.metric('Mean Predicted', f'{forecast_mean.mean():.1f} fraud/hour')
        f4.metric('Model MAE', f'{metadata.get('forecast_mae', 0):.4f}')

        ## forecast plot
        fig, axes = plt.subplots(2, 1, figsize = (14,9))

        axes[0].fill_between(hourly['Hour'], hourly['fraud_count'], color = LEGIT_COLOR, alpha = .4, label = 'Historical fraud volume')
        axes[0].plot(hourly['Hour'], hourly['fraud_count'], color = LEGIT_COLOR, linewidth = 1.5)
        axes[0].plot(future_hours, forecast_mean, color = FRAUD_COLOR, linewidth = 2.5, linestyle = '--', marker = 'D', markersize = 5, label = f'{horizon}hr Forecast')
        axes[0].fill_between(future_hours, lower, upper, color = FRAUD_COLOR, alpha = .15, label = '95% CI')
        axes[0].axvspan(future_hours[0], future_hours[-1], alpha = .05, color = FRAUD_COLOR)
        axes[0].set_ylabel('Fraud Count')
        axes[0].set_title(f'Fraud Volume -> Historical + {horizon}hr ARIMA Forecast', fontweight = 'bold')
        axes[0].legend()
        axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _,: f'{max(0, x):.0f}'))

        ## zoomed forecast plot
        zoom_start = max(0, hourly['Hour'].max() - 12)
        zoom_hist = hourly[hourly['Hour'] >= zoom_start]
        axes[1].plot(zoom_hist['Hour'], zoom_hist['fraud_count'], color = LEGIT_COLOR, linewidth=2, marker = 'o', markersize = 4, label = 'Recent Actuals')
        axes[1].plot(future_hours, forecast_mean, color = FRAUD_COLOR, linewidth = 2.5, linestyle = '--', marker = 'D', markersize = 6, label = 'Forecast')
        axes[1].fill_between(future_hours, lower, upper, color = FRAUD_COLOR, alpha = .2, label = '95% CI')
        axes[1].axvline(x = hourly['Hour'].max(), color = 'gray', linestyle = ':', linewidth = 1.5, label = 'Forecast start')
        axes[1].set_xlabel('Hour (since dataset start)')
        axes[1].set_ylabel('Fraud Count')
        axes[1].set_title('Zoomed View -> Recent History + Forecast Window', fontweight = 'bold')
        axes[1].legend()
        axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{max(0, x):.0f}'))

        plt.suptitle(f'ARIMA({p}, {d}, {q}) Fraud Volume Forecast, {horizon} Hour Horizon')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        ## forecast table
        st.subheader('Hourly Forecast Detail')
        forecast_table = pd.DataFrame({
            'Hour': future_hours,
            'Predicted Fraud': forecast_mean.round(2),
            'Lower 95% CI': lower.round(2),
            'Upper 95% CI': upper.round(2),
            'Alert': ['Spike Warning' if v > hourly['fraud_count'].mean() * 2 else 'Normal' for v in forecast_mean]
        })
        st.dataframe(forecast_table, use_container_width=True, hide_index=True)

        ## spike alert
        spike_hours = forecast_table[forecast_table['Alert'] == 'Spike Warning']

        if len(spike_hours) > 0:
            st.warning(f'{len(spike_hours)} hour(s) in the forecast window show predicted fraud volume exceeding 2x the historical avg.')
        else:
            st.success('No fraud predicted in the forecast window.')
        
        ## historical patterns
        st.divider()
        st.subheader('Historical Fraud Patterns')

        fig2, axes2 = plt.subplots(1, 2, figsize = (14,4))
        
        axes2[0].bar(hourly['Hour'], hourly['fraud_count'], color = FRAUD_COLOR, alpha = .7, edgecolor = 'white')
        axes2[0].set_xlabel('Hour')
        axes2[0].set_ylabel('Fraud Count')
        axes2[0].set_title('Fraud Count by Hour')

        axes2[1].bar(hourly['Hour'], hourly['fraud_rate'] * 100, color = NEUTRAL_COLOR, alpha = .7, edgecolor = 'white')
        axes2[1].set_xlabel('Hour')
        axes2[1].set_ylabel('Fraud Rate (%)')
        axes2[1].set_title('Fraud Rate (%) by Hour')

        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()