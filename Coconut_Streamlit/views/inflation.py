"""Study-aligned coconut supply, seasonality, inflation analysis and prediction."""
import warnings
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.model_selection import TimeSeriesSplit, KFold
from .common import table, chart, download_table

FEATURES = ["coconut_supply_kg", "harvested_area_ha", "year_index", "month_sin", "month_cos", "supply_lag_1", "supply_lag_12"]


def _monthly_panel(wb):
    d = wb.monthly.copy()
    if "inflation_pct" not in d or d.inflation_pct.notna().sum() == 0:
        return pd.DataFrame()
    inflation = (d.dropna(subset=["inflation_pct"]).groupby("date", as_index=False)
        .agg(inflation_pct=("inflation_pct", "median"), inflation_records=("inflation_pct", "count"),
             inflation_min=("inflation_pct", "min"), inflation_max=("inflation_pct", "max")))
    # Sum production across barangays; do not sum repeated area-wide inflation values.
    supply = (d.groupby("date", as_index=False)
        .agg(coconut_supply_kg=("harvest_kg", lambda s: s.sum(min_count=1)),
             harvested_area_ha=("area_ha", "sum"), barangays_with_harvest=("harvest_kg", "count")))
    panel = supply.merge(inflation, on="date", how="left").sort_values("date")
    panel["year"] = panel.date.dt.year
    panel["month"] = panel.date.dt.month
    panel["inflation_variation"] = panel.inflation_max - panel.inflation_min
    return panel


def _model_frame(panel):
    z = panel.sort_values("date").copy()
    z["year_index"] = z.date.dt.year - z.date.dt.year.min()
    z["month_sin"] = np.sin(2 * np.pi * z.month / 12)
    z["month_cos"] = np.cos(2 * np.pi * z.month / 12)
    z["supply_lag_1"] = z.coconut_supply_kg.shift(1)
    z["supply_lag_12"] = z.coconut_supply_kg.shift(12)
    return z.dropna(subset=FEATURES + ["inflation_pct"]).replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURES + ["inflation_pct"])


def _metrics(y, pred):
    return {"MAE": mean_absolute_error(y, pred), "MSE": mean_squared_error(y, pred),
            "RMSE": np.sqrt(mean_squared_error(y, pred)),
            "R²": r2_score(y, pred) if len(y) > 1 and np.var(y) > 1e-12 else np.nan}


def _make_models():
    return {
        "Multiple Linear Regression": make_pipeline(StandardScaler(), LinearRegression()),
        "Random Forest Regression": RandomForestRegressor(n_estimators=250, max_depth=5, min_samples_leaf=2,
            random_state=42, n_jobs=1),
        "Support Vector Regression": make_pipeline(StandardScaler(), SVR(C=10, epsilon=0.1, kernel="rbf")),
    }


def _run_comparison(frame):
    """Compare models on a chronological holdout plus two CV diagnostics."""
    n = len(frame)
    if n < 18:
        return None, None, "At least 18 complete monthly rows (after lag features) are recommended for a basic chronological comparison."
    split = max(1, int(np.floor(n * .8)))
    if split < 12 or n - split < 3:
        split = n - 3
    train, test = frame.iloc[:split], frame.iloc[split:]
    Xtr, ytr = train[FEATURES], train.inflation_pct
    Xte, yte = test[FEATURES], test.inflation_pct
    rows, fitted = [], {}
    model_specs = _make_models()

    # Final chronological holdout: this is the primary basis for selecting a model.
    for name, model in model_specs.items():
        try:
            model.fit(Xtr, ytr)
            pred = model.predict(Xte)
            rows.append({"Algorithm": name, **_metrics(yte, pred), "Test months": len(test)})
            fitted[name] = model
        except Exception as exc:
            rows.append({"Algorithm": name, "Error": str(exc)})

    # ARIMA is an inflation-history-only benchmark; it does not use coconut supply.
    try:
        from statsmodels.tsa.arima.model import ARIMA
        order = (1, 1, 1) if len(ytr) >= 15 else (1, 0, 0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            arima = ARIMA(ytr.astype(float), order=order).fit()
            arima_pred = np.asarray(arima.forecast(steps=len(test)))
        rows.append({"Algorithm": f"ARIMA{order} (inflation-only baseline)", **_metrics(yte, arima_pred), "Test months": len(test)})
        fitted["ARIMA baseline"] = arima
    except Exception as exc:
        rows.append({"Algorithm": "ARIMA baseline", "Error": f"Unavailable: {exc}"})

    result = pd.DataFrame(rows)
    good = result.dropna(subset=["MAE"]) if "MAE" in result else pd.DataFrame()
    if good.empty:
        return result, None, "No model could be evaluated on this dataset."
    good = good.copy().sort_values(["MAE", "RMSE"], ascending=True)
    good["Holdout rank"] = range(1, len(good) + 1)
    result = result.merge(good[["Algorithm", "Holdout rank"]], on="Algorithm", how="left")
    winner = good.iloc[0]["Algorithm"]

    # K-fold is included because the objective explicitly requests it. It is a
    # shuffled diagnostic only; chronological holdout/TimeSeriesSplit remain primary.
    cv_rows = []
    if len(train) >= 12:
        kfold = KFold(n_splits=min(5, max(2, len(train) // 5)), shuffle=True, random_state=42)
        tscv = TimeSeriesSplit(n_splits=min(3, max(2, len(train) // 6))) if len(train) >= 15 else None
        from sklearn.base import clone
        for name, estimator in model_specs.items():
            kfold_values, ts_values = [], []
            for a, b in kfold.split(Xtr):
                try:
                    m = clone(estimator)
                    m.fit(Xtr.iloc[a], ytr.iloc[a])
                    kfold_values.append(_metrics(ytr.iloc[b], m.predict(Xtr.iloc[b])))
                except Exception:
                    continue
            if tscv is not None:
                for a, b in tscv.split(Xtr):
                    if len(a) < 6:
                        continue
                    try:
                        m = clone(estimator)
                        m.fit(Xtr.iloc[a], ytr.iloc[a])
                        ts_values.append(_metrics(ytr.iloc[b], m.predict(Xtr.iloc[b])))
                    except Exception:
                        continue
            row = {"Algorithm": name, "K-fold folds": len(kfold_values), "Time-series folds": len(ts_values)}
            for metric in ["MAE", "MSE", "RMSE", "R²"]:
                row[f"K-fold mean {metric}"] = float(np.nanmean([v[metric] for v in kfold_values])) if kfold_values else np.nan
                row[f"Time-series CV mean {metric}"] = float(np.nanmean([v[metric] for v in ts_values])) if ts_values else np.nan
            cv_rows.append(row)
    cv = pd.DataFrame(cv_rows)
    if not cv.empty:
        result = result.merge(cv, on="Algorithm", how="left")
    return result, {"models": fitted, "winner": winner, "train": train, "test": test,
                    "cv": cv}, None


def render(wb):
    st.title("Coconut Supply and Local Inflation Analysis")
    st.write("Study objective: examine whether coconut production volume, harvested area, year, and seasonal patterns are associated with local inflation, then compare predictive algorithms.")
    panel = _monthly_panel(wb)
    if panel.empty:
        st.warning("No usable Inflation Rate (%) values were found. Upload verified monthly inflation observations to use this module.")
        st.caption("Production analysis and coconut forecasting remain available. This module never invents inflation values.")
        return
    valid = panel.dropna(subset=["inflation_pct"]).copy()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Monthly inflation observations", f"{len(valid):,}")
    c2.metric("Average recorded inflation", f"{valid.inflation_pct.mean():.2f}%")
    c3.metric("Latest recorded inflation", f"{valid.sort_values('date').iloc[-1].inflation_pct:.2f}%")
    c4.metric("Period covered", f"{valid.date.min():%Y}–{valid.date.max():%Y}")
    st.caption("Monthly inflation is represented once per month using the median when repeated across barangay rows. Coconut harvest is summed across barangays. Confirm that the inflation indicator's geographic coverage matches Rosario.")

    st.header("1. Historical inflation and yearly production effect")
    a, b = st.columns(2)
    with a:
        st.subheader("Inflation trend by month")
        chart(alt.Chart(valid).mark_line(point=True).encode(x=alt.X("date:T", title="Month"),
            y=alt.Y("inflation_pct:Q", title="Inflation rate (%)", scale=alt.Scale(zero=False)),
            tooltip=["date:T", alt.Tooltip("inflation_pct:Q", format=".2f")]).properties(height=300).interactive())
    with b:
        st.subheader("Annual coconut production and inflation")
        annual = panel.groupby("year", as_index=False).agg(
            annual_coconut_production_kg=("coconut_supply_kg", lambda s: s.sum(min_count=1)),
            average_inflation_pct=("inflation_pct", "mean"), months_with_harvest=("coconut_supply_kg", "count"),
            months_with_inflation=("inflation_pct", "count"), harvested_area_month_sum_ha=("harvested_area_ha", "sum"))
        annual["harvest_year_complete"] = annual.months_with_harvest.eq(12)
        annual["inflation_year_complete"] = annual.months_with_inflation.eq(12)
        annual_index = annual.copy()
        for col in ["annual_coconut_production_kg", "average_inflation_pct"]:
            baseline = annual_index[col].dropna().iloc[0] if annual_index[col].notna().any() else np.nan
            annual_index[col + "_index"] = annual_index[col] / baseline * 100 if pd.notna(baseline) and baseline != 0 else np.nan
        chart(alt.Chart(annual_index).transform_fold(["annual_coconut_production_kg_index", "average_inflation_pct_index"],
            as_=["Measure", "Index (first available year = 100)"]).mark_line(point=True).encode(
            x=alt.X("year:O", title="Year"), y=alt.Y("Index (first available year = 100):Q", title="Indexed value (base = 100)"),
            color="Measure:N", tooltip=["year:O", "Measure:N", "Index (first available year = 100):Q"]).properties(height=300).interactive())
    table(annual.round(3))
    st.caption("The chart indexes both measures to 100 in their first available year so their direction can be compared despite different units. The annual table reports actual kg and percent. Years with fewer than 12 recorded harvest months are flagged; do not compare incomplete years as if they were full years.")
    download_table(annual, "yearly_production_and_inflation.csv")

    st.header("2. Seasonality: harvest and inflation by month")
    seasonal = panel.groupby("month", as_index=False).agg(
        average_monthly_harvest_kg=("coconut_supply_kg", "mean"),
        average_inflation_pct=("inflation_pct", "mean"),
        harvest_observations=("coconut_supply_kg", "count"), inflation_observations=("inflation_pct", "count"))
    seasonal["month_name"] = pd.to_datetime(seasonal.month, format="%m").dt.month_name()
    s1, s2 = st.columns(2)
    with s1:
        chart(alt.Chart(seasonal).mark_bar().encode(x=alt.X("month_name:O", sort=list(pd.date_range("2024-01-01", periods=12, freq="MS").strftime("%B")), title="Month"),
            y=alt.Y("average_monthly_harvest_kg:Q", title="Average harvest (kg)"), tooltip=["month_name", "average_monthly_harvest_kg", "harvest_observations"]).properties(height=300))
    with s2:
        chart(alt.Chart(seasonal).mark_line(point=True).encode(x=alt.X("month_name:O", sort=list(pd.date_range("2024-01-01", periods=12, freq="MS").strftime("%B")), title="Month"),
            y=alt.Y("average_inflation_pct:Q", title="Average inflation (%)"), tooltip=["month_name", "average_inflation_pct", "inflation_observations"]).properties(height=300))
    table(seasonal.round(3))
    download_table(seasonal, "monthly_seasonality_summary.csv")

    st.header("3. Significant factors and association analysis")
    paired = valid.dropna(subset=["coconut_supply_kg", "harvested_area_ha"]).copy()
    p1, p2 = st.columns(2)
    with p1:
        chart(alt.Chart(paired).mark_circle(size=75, opacity=.65).encode(x=alt.X("coconut_supply_kg:Q", title="Coconut production (kg)"),
            y=alt.Y("inflation_pct:Q", title="Inflation rate (%)"), tooltip=["date:T", "coconut_supply_kg", "inflation_pct"]).properties(height=300).interactive())
    with p2:
        chart(alt.Chart(paired).mark_circle(size=75, opacity=.65).encode(x=alt.X("harvested_area_ha:Q", title="Reported harvested area (ha)"),
            y=alt.Y("inflation_pct:Q", title="Inflation rate (%)"), tooltip=["date:T", "harvested_area_ha", "inflation_pct"]).properties(height=300).interactive())
    corr = paired[["coconut_supply_kg", "harvested_area_ha", "year", "month", "inflation_pct"]].corr(method="spearman")
    st.subheader("Spearman correlation matrix")
    st.dataframe(corr.round(3), width="stretch")
    st.caption("Correlation indicates association, not proof of causation. Year can represent a time trend; month represents recurring seasonal patterns. Other factors not in the dataset may also affect inflation.")
    st.subheader("Regression-based factor assessment")
    reg_data = _model_frame(panel)
    if len(reg_data) >= 18:
        try:
            import statsmodels.api as sm
            Xreg = sm.add_constant(reg_data[FEATURES].astype(float), has_constant="add")
            # HAC standard errors account for some serial dependence in monthly observations.
            fit = sm.OLS(reg_data.inflation_pct.astype(float), Xreg).fit(
                cov_type="HAC", cov_kwds={"maxlags": min(3, max(1, len(reg_data)//10))})
            ci = fit.conf_int()
            factor_table = pd.DataFrame({"Factor": fit.params.index, "Coefficient": fit.params.values,
                "p-value (HAC)": fit.pvalues.values, "95% CI lower": ci.iloc[:, 0].values,
                "95% CI upper": ci.iloc[:, 1].values})
            factor_table = factor_table[factor_table.Factor != "const"].sort_values("p-value (HAC)")
            table(factor_table.round(5))
            st.caption("This is a multivariable associational regression. A p-value below 0.05 is a statistical signal under the model assumptions, not proof of causation. Interpret coefficients in their stated units and consider omitted economic variables.")
            st.metric("Regression R² (in-sample)", f"{fit.rsquared:.3f}")
        except Exception as exc:
            st.info(f"Regression factor assessment is unavailable for this dataset: {exc}")
    else:
        st.info("At least 18 complete monthly rows after lag preparation are recommended for regression-based factor assessment.")

    st.header("4. Inflation prediction and algorithm comparison")
    model_data = _model_frame(panel)
    st.write("Prediction target: monthly inflation rate (%). Supply and harvested-area variables are used with year trend, seasonal month features, and previous harvest values. The final test period is kept in chronological order to reduce time leakage.")
    if len(model_data) < 18:
        st.warning(f"The model comparison charts are not displayed because only {len(model_data)} complete monthly observations remain after preparing the 1-month and 12-month production lags. At least 18 are required by this dashboard's minimum comparison rule.")
        st.markdown("**Why the charts are missing:** the comparison is conditional on having enough complete monthly data to train and test the models. Missing inflation values or missing harvest records can reduce the usable row count.")
        st.caption("To enable the comparison, provide a continuous monthly series with inflation rate, coconut harvest, and harvested area. Do not fill missing inflation values with guesses. The dashboard will show the model table and charts once enough complete observations are available.")
    else:
        metrics, result, error = _run_comparison(model_data)
        if error:
            st.warning(error)
        else:
            st.subheader("Model comparison dashboard")
            st.caption("All algorithms are evaluated against the same final chronological test period. Rank is based on holdout MAE, with RMSE used as a tie-breaker.")
            ranked = metrics.sort_values("Holdout rank").copy()
            top = ranked.iloc[0]
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Best model (holdout)", str(top["Algorithm"]).replace(" (inflation-only baseline)", ""))
            m2.metric("Best MAE", f"{top['MAE']:.4f}")
            m3.metric("Best RMSE", f"{top['RMSE']:.4f}")
            m4.metric("Best R²", f"{top['R²']:.4f}" if pd.notna(top["R²"]) else "Not available")
            st.markdown("#### Side-by-side performance results")
            display_cols = [c for c in ["Holdout rank", "Algorithm", "MAE", "MSE", "RMSE", "R²", "Test months"] if c in ranked]
            table(ranked[display_cols].round(4))
            download_table(ranked, "inflation_algorithm_comparison.csv")
            # Use Streamlit's native chart renderer for robust display on Streamlit
            # Community Cloud; this avoids charts silently failing because of
            # Altair version/schema differences.
            st.markdown("#### Prediction error comparison (lower is better)")
            error_plot = ranked.set_index("Algorithm")[["MAE", "RMSE"]].apply(pd.to_numeric, errors="coerce")
            error_plot = error_plot.replace([np.inf, -np.inf], np.nan).dropna(how="all")
            if not error_plot.empty:
                st.bar_chart(error_plot, horizontal=True, color=["#8B5CF6", "#C4B5FD"], height=330)
                st.caption("Each model has two bars: MAE and RMSE. Shorter bars indicate smaller prediction errors.")
            else:
                st.warning("The error comparison chart cannot be drawn because all MAE/RMSE values are missing.")

            st.markdown("#### R² comparison (higher is better)")
            r2_plot = ranked.set_index("Algorithm")[["R²"]].apply(pd.to_numeric, errors="coerce")
            r2_plot = r2_plot.replace([np.inf, -np.inf], np.nan).dropna()
            if not r2_plot.empty:
                st.bar_chart(r2_plot, horizontal=True, color="#6D28D9", height=260)
                st.caption("R² closer to 1 generally indicates a better fit. Negative values are possible and should be interpreted as poor out-of-sample performance.")
            else:
                st.info("R² is unavailable for this test period (for example, if actual inflation values are constant). The MAE, MSE, and RMSE comparison remains available.")
            st.success(f"Best model for this dataset and test period: {result['winner']}")
            st.caption("MAE, MSE, and RMSE: lower is better. R²: closer to 1 is generally better; a negative score means the model performed worse than predicting the test-set mean. ARIMA is an inflation-history-only baseline and does not estimate coconut-supply effects.")
            if not result["cv"].empty:
                st.markdown("#### Cross-validation comparison")
                st.write("K-fold CV is shown as a supplementary shuffled-fold diagnostic. Time-series CV uses expanding training windows and preserves chronological order; prioritize it over shuffled K-fold when judging time-series reliability.")
                cv_cols = [c for c in result["cv"].columns if c == "Algorithm" or c.endswith("folds") or c.endswith("mean MAE") or c.endswith("mean MSE") or c.endswith("mean RMSE") or c.endswith("mean R²")]
                table(result["cv"][cv_cols].round(4))
                download_table(result["cv"], "inflation_cross_validation_comparison.csv")
            else:
                st.info("There are not enough training observations to run the requested cross-validation reliably. Add more monthly history.")
            chosen = st.selectbox("View actual versus predicted inflation", [x for x in result["models"] if x != "ARIMA baseline"], key="inflation_model_view")
            model = result["models"][chosen]
            test = result["test"].copy()
            test["predicted_inflation_pct"] = model.predict(test[FEATURES])
            plot = test[["date", "inflation_pct", "predicted_inflation_pct"]].melt("date", var_name="Series", value_name="Inflation (%)")
            chart(alt.Chart(plot).mark_line(point=True).encode(x="date:T", y=alt.Y("Inflation (%):Q", title="Inflation (%)"), color="Series:N", tooltip=["date:T", "Series:N", alt.Tooltip("Inflation (%):Q", format=".3f")]).properties(height=320).interactive())
            table(test[["date", "inflation_pct", "predicted_inflation_pct"]].round(4))
            download_table(test[["date", "inflation_pct", "predicted_inflation_pct"]], "actual_vs_predicted_inflation.csv")

            st.subheader("Future 12-month scenario prediction")
            st.info("This is a what-if scenario, not a guaranteed forecast. Enter an assumed average monthly harvest and harvested area. The model uses these assumptions to estimate possible inflation values.")
            latest = panel.sort_values("date").iloc[-1]
            c1, c2 = st.columns(2)
            with c1:
                assumed_supply = st.number_input("Assumed monthly coconut harvest (kg)", min_value=0.0,
                    value=float(max(0, latest.coconut_supply_kg if pd.notna(latest.coconut_supply_kg) else 0)), step=1000.0)
            with c2:
                assumed_area = st.number_input("Assumed monthly reported harvested area (ha)", min_value=0.0,
                    value=float(max(0, latest.harvested_area_ha if pd.notna(latest.harvested_area_ha) else 0)), step=1.0)
            if st.button("Generate 12-month inflation scenario", type="primary", key="make_inflation_scenario"):
                hist = panel.sort_values("date").copy()
                future_dates = pd.date_range(hist.date.max() + pd.offsets.MonthBegin(1), periods=12, freq="MS")
                supplies = hist.coconut_supply_kg.dropna().tolist()
                yprev = float(hist.inflation_pct.dropna().iloc[-1])
                future_rows = []
                for dt in future_dates:
                    supplies.append(float(assumed_supply))
                    row = {"coconut_supply_kg": float(assumed_supply), "harvested_area_ha": float(assumed_area),
                           "year_index": dt.year - int(panel.date.dt.year.min()),
                           "month_sin": np.sin(2*np.pi*dt.month/12), "month_cos": np.cos(2*np.pi*dt.month/12),
                           "supply_lag_1": supplies[-2] if len(supplies) >= 2 else float(assumed_supply),
                           "supply_lag_12": supplies[-13] if len(supplies) >= 13 else float(assumed_supply)}
                    future_rows.append(row)
                future_x = pd.DataFrame(future_rows)[FEATURES]
                forecast = model.predict(future_x)
                future = pd.DataFrame({"date": future_dates, "predicted_inflation_pct": forecast,
                                       "assumed_monthly_harvest_kg": assumed_supply,
                                       "assumed_harvested_area_ha": assumed_area})
                st.session_state["inflation_scenario_result"] = future
            if "inflation_scenario_result" in st.session_state:
                future = st.session_state["inflation_scenario_result"]
                chart(alt.Chart(future).mark_line(point=True).encode(x="date:T", y=alt.Y("predicted_inflation_pct:Q", title="Predicted inflation (%)"), tooltip=["date:T", "predicted_inflation_pct"]).properties(height=300))
                table(future.round(4))
                download_table(future, "future_inflation_scenario.csv")

    with st.expander("Data quality and interpretation notes"):
        quality = valid[["date", "inflation_records", "inflation_min", "inflation_max", "inflation_variation"]].copy()
        table(quality.round(4))
        inconsistent = quality[quality.inflation_variation.abs() > 1e-9]
        if not inconsistent.empty:
            st.warning(f"{len(inconsistent)} months have different inflation values across source rows. Verify the source and geographic coverage before interpreting the median as local inflation.")
        else:
            st.success("No within-month differences were detected among the repeated inflation values.")
        st.caption("A statistically significant factor requires an appropriate inferential model and adequate data. Predictive feature importance or correlation alone should not be described as statistical significance or causal impact.")
        download_table(panel, "monthly_coconut_supply_inflation.csv")
