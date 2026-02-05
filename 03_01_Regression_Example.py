import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import ExtraTreesRegressor


from sklearn.model_selection import KFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import make_scorer


# Definir una función para eliminar outliers usando el rango intercuartílico (IQR)
def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)  # Primer cuartil (25%)
    Q3 = df[column].quantile(0.75)  # Tercer cuartil (75%)
    IQR = Q3 - Q1  # Rango intercuartílico
    lower_bound = Q1 - 1.5 * IQR  # Límite inferior
    upper_bound = Q3 + 1.5 * IQR  # Límite superior
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

# Crear gráficos de dispersión antes de la limpieza
def scatter_plots_comparison(data_before, data_after, target_column):
    import matplotlib
    matplotlib.use('TkAgg')
    plt.ion()
    numeric_columns = [
        col for col in data_before.columns
        if col != target_column and data_before[col].dtype in ['float64', 'int64']
    ]

    n_cols = 2  # Número de columnas en el grid de subplots
    n_rows = (len(numeric_columns) + 1) // n_cols  # Número de filas, ajustado al total de columnas
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axes = axes.flatten()  # Aplanar la matriz de subplots para un acceso sencillo

    for i, column in enumerate(numeric_columns):
        ax = axes[i]
        # Graficar los datos antes de la limpieza
        ax.scatter(data_before[column], data_before[target_column], alpha=0.4, label='Antes', color='blue')
        # Graficar los datos después de la limpieza
        ax.scatter(data_after[column], data_after[target_column], alpha=0.4, label='Después', color='orange')

        # Configurar títulos y etiquetas
        ax.set_title(f"{column} vs {target_column}")
        ax.set_xlabel(column)
        ax.set_ylabel(target_column)
        ax.grid(True)
        ax.legend()

    # Eliminar ejes sobrantes si no se usan
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    # Ajustar el layout
    plt.show()

# Crear una función para calcular la matriz de p-values
def calculate_pvalues(data):
    # H0: no existe correlación lineal entre vbles.
    # H1: existe correlación lineal entre vbles.
    # Si p-value<5%, se rechaza la H0, hay evidencia estadística que confirma la correlación.
    # Si p-value>5%, BO se rechaza la H0, NO hay evidencia estadística que confirma la correlación.

    cols = data.columns
    pvalues = pd.DataFrame(index=cols, columns=cols)  # Crear un DataFrame vacío
    for col1 in cols:
        for col2 in cols:
            if col1 == col2:
                pvalues.loc[col1, col2] = 0  # El p-value de una variable consigo misma es 0
            else:
                _, p_value = pearsonr(data[col1], data[col2])  # Calcular p-value
                pvalues.loc[col1, col2] = p_value
    return pvalues.astype(float)

# Función para calcular R2 ajustado
def r2_adjusted(y_true, y_pred, n, p):
    r2 = r2_score(y_true, y_pred)
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)

def main():

    ####################################################################################################################
    ############################################### CARGAR DATOS #######################################################
    ####################################################################################################################

    # DATASET: https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength

    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls'
    data = pd.read_excel(url)

    # Mostrar las primeras filas del conjunto de datos
    print(data.head())
    # Obtener información sobre las columnas y tipos de datos
    print(data.info())
    # Describir las estadísticas básicas de las variables numéricas
    print(data.describe())

    target_vble = 'Concrete compressive strength(MPa, megapascals) '

    ####################################################################################################################
    ################################################### OUTLIERS #######################################################
    ####################################################################################################################

    # Aplicar la eliminación de outliers a cada columna numérica
    for column in data.columns:
        if data[column].dtype in ['float64', 'int64']:  # Solo aplicar a columnas numéricas
            data_filt = remove_outliers_iqr(data, column)

    # Revisar el tamaño del conjunto de datos después de eliminar outliers
    print(f"El tamaño del conjunto de datos después de eliminar outliers es: {data.shape}")

    # Llamar a la función antes de eliminar outliers
    scatter_plots_comparison(data, data_filt, target_vble)

    ####################################################################################################################
    ################################################### CORRELACIÓN ####################################################
    ####################################################################################################################

    # Calcular la matriz de correlación
    correlation_matrix = data_filt.corr()
    pvalues_matrix = calculate_pvalues(data_filt)

    # Crear subplots para visualizar los mapas de calor
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))  # Dos gráficos en una fila
    # Mapa de calor de la matriz de correlación
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, annot_kws={"size": 10}, ax=axes[0])
    axes[0].set_title("Mapa de calor: Correlación")
    # Mapa de calor de la matriz de p-values
    sns.heatmap(pvalues_matrix, annot=True, fmt=".4f", cmap="coolwarm", cbar=True, annot_kws={"size": 10}, ax=axes[1])
    axes[1].set_title("Mapa de calor: p-values")
    # Ajustar layout
    plt.tight_layout()
    plt.show()

    # Umbral de correlación
    threshold = 0.75
    # Identificar pares de variables con correlación alta
    high_correlation_pairs = [
        (var1, var2, correlation_matrix.loc[var1, var2])
        for var1 in correlation_matrix.columns
        for var2 in correlation_matrix.columns
        if var1 != var2 and abs(correlation_matrix.loc[var1, var2]) > threshold
    ]
    # Mostrar los pares con correlación alta
    print("Pares de variables con alta correlación:")
    for var1, var2, corr_value in high_correlation_pairs:
        print(f"{var1} y {var2}: Correlación = {corr_value:.2f}")

    ####################################################################################################################
    ############################################### EJECUCIÓN DE MODELOS ###############################################
    ####################################################################################################################

    # Inicializar los modelos
    models = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Extra Trees': ExtraTreesRegressor(n_estimators=100, max_depth=5),
            'GBR': GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3),
            'ADA': AdaBoostRegressor(n_estimators=100, learning_rate=0.1)
             }

    # Separar las características (X) y la variable objetivo (y)
    X = data.drop(columns=[target_vble])  # Sustituir por el nombre de la variable objetivo
    y = data[target_vble]
    # Dividir en entrenamiento (80%) y prueba (20%). La prueba la dejamos out-of-the-box. Se realizará la validación cruzada con el set de training.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Función para calcular R2 ajustado
    def r2_adjusted(y_true, y_pred, n, p):
        r2 = r2_score(y_true, y_pred)
        return 1 - (1 - r2) * (n - 1) / (n - p - 1)

    # Definir K-Fold
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)

    print('############# VALIDACIÓN CRUZADA  (K-Fold con Train Set) ###############')
    # Evaluar cada modelo
    results = {}
    for name, model in models.items():
        print(f"Evaluando modelo: {name}")

        # Variables para ajustar el R2
        n = len(y_train)  # Número de observaciones
        p = X_train.shape[1]  # Número de predictores (features)

        # Cross-validation para RMSE
        neg_mse_scores = cross_val_score(model, X_train, y_train, cv=kfold, scoring='neg_mean_squared_error')
        rmse_scores = np.sqrt(-neg_mse_scores)

        # Cross-validation para R2 ajustado
        r2_scores = cross_val_score(model, X_train, y_train, cv=kfold, scoring=make_scorer(r2_adjusted, n=n, p=p))

        # Guardar resultados
        results[name] = {
            'RMSE Mean': rmse_scores.mean(),
            'RMSE Std': rmse_scores.std(),
            'R2 Adjusted Mean': r2_scores.mean(),
            'R2 Adjusted Std': r2_scores.std()
        }

        # Mostrar resultados del modelo
        print(
            f"{name} - RMSE: {rmse_scores.mean():.4f} ± {rmse_scores.std():.4f}, R2 Ajustado: {r2_scores.mean():.4f} ± {r2_scores.std():.4f}")

    print('############# VALIDACIÓN CRUZADA  (Test Set) ###############')
    # Entrenar y evaluar en el conjunto de prueba
    for name, model in models.items():
        # Entrenar el modelo
        model.fit(X_train, y_train)

        # Predicciones en el conjunto de prueba
        y_pred = model.predict(X_test)

        # Calcular métricas
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        # Calcular R2 ajustado en el conjunto de prueba
        test_r2_adjusted = r2_adjusted(y_test, y_pred, len(y_test), X_test.shape[1])

        print(f"Resultados en Test para {name}:")
        print(f"RMSE: {test_rmse:.4f}, R2 Ajustado: {test_r2_adjusted:.4f}\n")

    ####################################################################################################################
    ############################################### AJUSTE HIPERPARÁMETROS #############################################
    ####################################################################################################################

    # Definir el modelo GBR
    gbr_model = GradientBoostingRegressor()

    # Definir el espacio de hiperparámetros para la búsqueda
    param_grid = {
                    'n_estimators': [50, 100, 150, 200],
                    'learning_rate': [0.01, 0.05, 0.1, 0.2],
                    'max_depth': [3, 4, 5, 6]
    }

    # # Definir la validación cruzada con k-fold
    # grid_search = GridSearchCV(estimator=gbr_model, param_grid=param_grid,
    #                            cv=5, scoring='neg_mean_squared_error',
    #                            n_jobs=-1, verbose=2)
    #
    # # Ajustar el modelo con la búsqueda de cuadrícula
    # grid_search.fit(X_train, y_train)
    # print(f"Mejores parámetros encontrados: {grid_search.best_params_}")
    # print(f"Mejor puntuación (neg_MSE): {grid_search.best_score_}")

    # Definir la validación cruzada con k-fold
    random_search = RandomizedSearchCV(estimator=gbr_model, param_distributions=param_grid,
                                       n_iter=50, cv=5, scoring='neg_mean_squared_error',
                                       n_jobs=-1, verbose=2, random_state=42)

    # Ajustar el modelo con la búsqueda aleatoria
    random_search.fit(X_train, y_train)

    # Ver los mejores parámetros
    print(f"Mejores parámetros encontrados: {random_search.best_params_}")

    # Ver el mejor rendimiento
    print(f"Mejor puntuación (neg_MSE): {random_search.best_score_}")
    print(f"Mejor puntuación (RMSE): {np.sqrt(-random_search.best_score_)}")

    ###################################################################################################################
    #################################################### MODELO FINAL #################################################
    ###################################################################################################################

    gbr_model_final = GradientBoostingRegressor(n_estimators = random_search.best_params_['n_estimators'],
                                                learning_rate = random_search.best_params_['learning_rate'],
                                                max_depth = random_search.best_params_['max_depth'])
    # gbr_model_final = GradientBoostingRegressor(n_estimators = 200,
    #                                             learning_rate = 0.1,
    #                                             max_depth = 4)
    # Entrenar el modelo
    gbr_model_final.fit(X_train, y_train)
    # Predicciones en el conjunto de prueba
    y_pred = gbr_model_final.predict(X_test)
    # Calcular métricas
    final_test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    # Calcular R2 ajustado en el conjunto de prueba
    final_test_r2_adjusted = r2_adjusted(y_test, y_pred, len(y_test), X_test.shape[1])
    print(f"****** RESULTADOS DE VALIDACIÓN MODELO FINAL *******")
    print(f'RMSE (MPa): {final_test_rmse}.2f')
    print(f'R2 (-): {final_test_r2_adjusted}.2f')

    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.6, color="b", edgecolors="k")
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color="red", linestyle="--", linewidth=2)
    plt.title("Comparación: Valores Reales vs Predichos")
    plt.xlabel("Valores Reales")
    plt.ylabel("Valores Predichos")
    plt.grid(True)
    plt.show()

    residuals = y_test - y_pred

    plt.figure(figsize=(8, 6))
    plt.scatter(y_pred, residuals, alpha=0.6, color="purple", edgecolors="k")
    plt.axhline(0, color="red", linestyle="--", linewidth=2)
    plt.title("Gráfico de Residuos")
    plt.xlabel("Valores Predichos")
    plt.ylabel("Residuos (Reales - Predichos)")
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(8, 6))
    sns.histplot(residuals, kde=True, color="orange", bins=30)
    plt.axvline(0, color="red", linestyle="--", linewidth=2)
    plt.title("Distribución de Residuos")
    plt.xlabel("Residuos")
    plt.ylabel("Frecuencia")
    plt.grid(True)
    plt.show()

    print("Main function executed")

if __name__ == "__main__":
    main()