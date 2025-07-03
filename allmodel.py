import pandas as pd
import numpy as np
import os
from glob import glob
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.font_manager as fm

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

folder_path = "취업률예측"
file_paths = glob(os.path.join(folder_path, "*.xlsx"))
df_list = [pd.read_excel(fp, sheet_name=None, engine='openpyxl') for fp in file_paths]
df = pd.concat([sheet for file in df_list for sheet in file.values()], ignore_index=True)

df = df.dropna(subset=['대학교명', '학과명', '모집인원', '평균입학등급', '취업률'])

df['대학교명'] = df['대학교명'].replace('국립한밭대학교', 'H대학교')
unique_univs = df['대학교명'].unique()
anon_map = {univ: f"{chr(65+i)}대학교" for i, univ in enumerate(unique_univs) if univ != 'H대학교'}
df['대학교명'] = df['대학교명'].replace(anon_map)

X = df[['대학교명', '학과명', '모집인원', '평균입학등급']]
y = df['취업률']

categorical = ['대학교명', '학과명']
numeric = ['모집인원', '평균입학등급']

dept_categories = sorted(df['학과명'].astype(str).unique().tolist())
univ_categories = sorted(df['대학교명'].astype(str).unique().tolist())

if '산업경영공학과' in dept_categories:
    dept_categories.remove('산업경영공학과')
    dept_categories = ['산업경영공학과'] + dept_categories
if 'H대학교' in univ_categories:
    univ_categories.remove('H대학교')
    univ_categories = ['H대학교'] + univ_categories

onehot_encoder = OneHotEncoder(categories=[univ_categories, dept_categories], drop='first', handle_unknown='ignore')
preprocessor = ColumnTransformer([
    ('cat', onehot_encoder, categorical),
    ('num', StandardScaler(), numeric)
])

models = {
    "LinearRegression": (LinearRegression(), {}),
    "RandomForest": (RandomForestRegressor(random_state=42), {
        'regressor__n_estimators': [100],
        'regressor__max_depth': [5, 10, None]
    }),
    "XGBoost": (XGBRegressor(objective='reg:squarederror', random_state=42), {
        'regressor__n_estimators': [100],
        'regressor__max_depth': [3, 5],
        'regressor__learning_rate': [0.05, 0.1]
    }),
    "LightGBM": (LGBMRegressor(random_state=42), {
        'regressor__n_estimators': [100],
        'regressor__num_leaves': [31, 50],
        'regressor__learning_rate': [0.05, 0.1]
    })
}

hanbat_df = df[df['대학교명'] == 'H대학교']

for name, (regressor, param_grid) in models.items():
    print(f"\n===== 🔍 모델: {name} =====")
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', regressor)
    ])

    if param_grid:
        grid = GridSearchCV(pipeline, param_grid, cv=5, scoring='r2', n_jobs=-1)
        grid.fit(X, y)
        model = grid.best_estimator_
        best_params = grid.best_params_
    else:
        pipeline.fit(X, y)
        model = pipeline
        best_params = {}

    y_pred_all = model.predict(X)
    df_result = df.copy()
    df_result['예측 취업률'] = y_pred_all
    df_result['예측 오차'] = df_result['예측 취업률'] - df_result['취업률']
    hanbat_result = df_result[df_result['대학교명'] == 'H대학교']

    r2 = r2_score(hanbat_result['취업률'], hanbat_result['예측 취업률'])
    mae = mean_absolute_error(hanbat_result['취업률'], hanbat_result['예측 취업률'])
    rmse = np.sqrt(mean_squared_error(hanbat_result['취업률'], hanbat_result['예측 취업률']))

    print("▶ 한밭대 예측 R²:", r2)
    print("▶ MAE:", mae)
    print("▶ RMSE:", rmse)
    print("▶ Best Params:", best_params)

    cv_metrics = ['r2', 'neg_mean_absolute_error', 'neg_root_mean_squared_error']
    cv_results = cross_validate(model, X, y, cv=5, scoring=cv_metrics)
    r2_scores = cv_results['test_r2']
    mae_scores = -cv_results['test_neg_mean_absolute_error']
    rmse_scores = -cv_results['test_neg_root_mean_squared_error']

    print("▶ 5-Fold 평균 R²:", np.mean(r2_scores))
    print("▶ 5-Fold 평균 MAE:", np.mean(mae_scores))
    print("▶ 5-Fold 평균 RMSE:", np.mean(rmse_scores))

    output_dir = os.path.join(folder_path, f"{name}_H대학교_예측결과")
    os.makedirs(output_dir, exist_ok=True)

    hanbat_result.to_csv(os.path.join(output_dir, "예측결과.csv"), index=False, encoding='utf-8-sig')

    cv_df = pd.DataFrame({
        'Fold': [f"Fold{i+1}" for i in range(5)],
        'R²': r2_scores,
        'MAE': mae_scores,
        'RMSE': rmse_scores
    })
    cv_df.loc['평균'] = cv_df.mean(numeric_only=True)
    cv_df.loc['표준편차'] = cv_df.std(numeric_only=True)
    cv_df.to_csv(os.path.join(output_dir, "CV_5Fold_전체지표.csv"), index=True, encoding='utf-8-sig')

    if hasattr(model.named_steps['regressor'], 'feature_importances_'):
        importances = model.named_steps['regressor'].feature_importances_
    else:
        importances = model.named_steps['regressor'].coef_

    feature_names_cat = model.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(categorical)
    feature_names_all = np.concatenate([feature_names_cat, numeric])

    importance_col = 'Importance' if name != "LinearRegression" else 'Coefficient'
    importance_df = pd.DataFrame({
        'Feature': feature_names_all,
        importance_col: importances
    }).sort_values(by=importance_col, key=np.abs, ascending=False)

    importance_df.to_csv(os.path.join(output_dir, "변수중요도.csv"), index=False, encoding='utf-8-sig')

    if name == "LinearRegression":
        print("\n 회귀계수 상위 변수:")
        print(importance_df.head(10))
        importance_df.to_csv(os.path.join(output_dir, "회귀계수_해석.csv"), index=False, encoding='utf-8-sig')

    plt.figure(figsize=(10, 6))
    plt.barh(importance_df['Feature'][:20][::-1], importance_df[importance_col][:20][::-1])
    plt.xlabel("중요도" if name != "LinearRegression" else "계수 크기")
    plt.title(f"{name} 상위 20개 변수 중요도")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "변수중요도_시각화.png"))
    plt.close()

plt.figure(figsize=(14, 6))
sns.boxplot(x='학과명', y='취업률', data=df)
plt.xticks(rotation=90)
plt.title("학과별 취업률 분포 (차이 강조)")
plt.tight_layout()
plt.savefig(os.path.join(folder_path, "학과별_취업률_분포.png"))
plt.close()
