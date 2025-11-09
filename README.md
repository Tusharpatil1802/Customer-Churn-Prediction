# Customer-Churn-Prediction

{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "da33ecf2-83aa-4f07-8a39-e53b8069a81c",
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "\n",
    "from sklearn.model_selection import train_test_split, GridSearchCV\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "from sklearn.metrics import classification_report, confusion_matrix, accuracy_score\n",
    "from sklearn.ensemble import RandomForestClassifier\n",
    "from imblearn.over_sampling import SMOTE"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "dd2adcbe-f39c-4c5d-ab74-33688e070ffe",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      " Data Sample:\n"
     ]
    },
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>age</th>\n",
       "      <th>tenure</th>\n",
       "      <th>monthly_charges</th>\n",
       "      <th>total_charges</th>\n",
       "      <th>internet_service</th>\n",
       "      <th>contract</th>\n",
       "      <th>payment_method</th>\n",
       "      <th>gender</th>\n",
       "      <th>churn</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>56</td>\n",
       "      <td>3</td>\n",
       "      <td>44.606966</td>\n",
       "      <td>215.205673</td>\n",
       "      <td>Fiber optic</td>\n",
       "      <td>Month-to-month</td>\n",
       "      <td>Credit card</td>\n",
       "      <td>Male</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>69</td>\n",
       "      <td>3</td>\n",
       "      <td>67.669548</td>\n",
       "      <td>3272.951895</td>\n",
       "      <td>None</td>\n",
       "      <td>Month-to-month</td>\n",
       "      <td>Bank transfer</td>\n",
       "      <td>Female</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>46</td>\n",
       "      <td>9</td>\n",
       "      <td>59.436532</td>\n",
       "      <td>1632.487579</td>\n",
       "      <td>Fiber optic</td>\n",
       "      <td>Month-to-month</td>\n",
       "      <td>Bank transfer</td>\n",
       "      <td>Female</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>32</td>\n",
       "      <td>7</td>\n",
       "      <td>74.941424</td>\n",
       "      <td>1353.835466</td>\n",
       "      <td>DSL</td>\n",
       "      <td>Month-to-month</td>\n",
       "      <td>Credit card</td>\n",
       "      <td>Female</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>60</td>\n",
       "      <td>5</td>\n",
       "      <td>94.687572</td>\n",
       "      <td>1697.115584</td>\n",
       "      <td>DSL</td>\n",
       "      <td>Month-to-month</td>\n",
       "      <td>Electronic check</td>\n",
       "      <td>Male</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   age  tenure  monthly_charges  total_charges internet_service  \\\n",
       "0   56       3        44.606966     215.205673      Fiber optic   \n",
       "1   69       3        67.669548    3272.951895             None   \n",
       "2   46       9        59.436532    1632.487579      Fiber optic   \n",
       "3   32       7        74.941424    1353.835466              DSL   \n",
       "4   60       5        94.687572    1697.115584              DSL   \n",
       "\n",
       "         contract    payment_method  gender  churn  \n",
       "0  Month-to-month       Credit card    Male      0  \n",
       "1  Month-to-month     Bank transfer  Female      0  \n",
       "2  Month-to-month     Bank transfer  Female      0  \n",
       "3  Month-to-month       Credit card  Female      0  \n",
       "4  Month-to-month  Electronic check    Male      0  "
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "np.random.seed(42)\n",
    "n_samples = 1000\n",
    "\n",
    "data = pd.DataFrame({\n",
    "    'age': np.random.randint(18, 70, size=n_samples),\n",
    "    'tenure': np.random.randint(1, 10, size=n_samples),\n",
    "    'monthly_charges': np.random.uniform(20, 120, size=n_samples),\n",
    "    'total_charges': np.random.uniform(200, 5000, size=n_samples),\n",
    "    'internet_service': np.random.choice(['DSL', 'Fiber optic', 'None'], size=n_samples, p=[0.4, 0.5, 0.1]),\n",
    "    'contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], size=n_samples, p=[0.6, 0.25, 0.15]),\n",
    "    'payment_method': np.random.choice(['Credit card', 'Bank transfer', 'Electronic check'], size=n_samples),\n",
    "    'gender': np.random.choice(['Male', 'Female'], size=n_samples),\n",
    "})\n",
    "\n",
    "# Generate churn target with some dependency on features\n",
    "data['churn'] = np.where(\n",
    "    (data['tenure'] < 3) & (data['contract'] == 'Month-to-month') |\n",
    "    (data['internet_service'] == 'Fiber optic') & (data['monthly_charges'] > 90),\n",
    "    1, 0\n",
    ")\n",
    "\n",
    "print(\" Data Sample:\")\n",
    "display(data.head())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "2bec03bb-f64e-4b09-b311-08f22b219f77",
   "metadata": {},
   "outputs": [],
   "source": [
    "data_encoded = pd.get_dummies(data, drop_first=True)\n",
    "\n",
    "# Split features and target\n",
    "X = data_encoded.drop('churn', axis=1)\n",
    "y = data_encoded['churn']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "26eef6b1-9870-4b9f-a706-4b0288ba5ba6",
   "metadata": {},
   "outputs": [],
   "source": [
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "9f5ac32b-8c9d-488e-ac7f-2aac008d9196",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Before SMOTE: {0: 591, 1: 209}\n",
      "After SMOTE:  {0: 591, 1: 591}\n"
     ]
    }
   ],
   "source": [
    "sm = SMOTE(random_state=42)\n",
    "X_train_res, y_train_res = sm.fit_resample(X_train, y_train)\n",
    "\n",
    "print(f\"Before SMOTE: {y_train.value_counts().to_dict()}\")\n",
    "print(f\"After SMOTE:  {y_train_res.value_counts().to_dict()}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "c1d9e481-d807-4aef-9088-7ae57b0da58f",
   "metadata": {},
   "outputs": [],
   "source": [
    "scaler = StandardScaler()\n",
    "X_train_scaled = scaler.fit_transform(X_train_res)\n",
    "X_test_scaled = scaler.transform(X_test)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "7aac3c39-bf2a-4cc3-9e27-1e66185db2d4",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      " Best Parameters: {'max_depth': 15, 'min_samples_split': 2, 'n_estimators': 100}\n"
     ]
    }
   ],
   "source": [
    "rf = RandomForestClassifier(random_state=42)\n",
    "\n",
    "param_grid = {\n",
    "    'n_estimators': [100, 200],\n",
    "    'max_depth': [5, 10, 15],\n",
    "    'min_samples_split': [2, 5]\n",
    "}\n",
    "\n",
    "grid = GridSearchCV(rf, param_grid, cv=3, scoring='accuracy', n_jobs=-1)\n",
    "grid.fit(X_train_scaled, y_train_res)\n",
    "\n",
    "print(\" Best Parameters:\", grid.best_params_)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "8a24be1d-d882-4809-8207-b32ba41d6821",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      " Classification Report:\n",
      "               precision    recall  f1-score   support\n",
      "\n",
      "           0       1.00      0.95      0.98       148\n",
      "           1       0.88      1.00      0.94        52\n",
      "\n",
      "    accuracy                           0.96       200\n",
      "   macro avg       0.94      0.98      0.96       200\n",
      "weighted avg       0.97      0.96      0.97       200\n",
      "\n",
      "Accuracy: 0.965\n"
     ]
    }
   ],
   "source": [
    "best_rf = grid.best_estimator_\n",
    "y_pred = best_rf.predict(X_test_scaled)\n",
    "\n",
    "print(\"\\n Classification Report:\\n\", classification_report(y_test, y_pred))\n",
    "print(\"Accuracy:\", accuracy_score(y_test, y_pred))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "cdb2d7e5-3800-452a-b780-a6469bf51b2c",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAf4AAAGHCAYAAABRQjAsAAAAOnRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjEwLjAsIGh0dHBzOi8vbWF0cGxvdGxpYi5vcmcvlHJYcgAAAAlwSFlzAAAPYQAAD2EBqD+naQAANkdJREFUeJzt3XlcVdX+//H3EfEIKCSYB/FiqVE5XUUtr5SJORSa5q9BSystNVMbyDHyOmQ3SG5ftcQhTcU0076lXuuWSWkOoYUDldrNLMwGz0XNHBARYf/+6OH5dgQNDueAuF7PHvvx6Kw9fba3+3i71l57b5tlWZYAAIARqlR0AQAAoPwQ/AAAGITgBwDAIAQ/AAAGIfgBADAIwQ8AgEEIfgAADELwAwBgEIIfAACDEPyoVL788ks9/PDDatCggapXr64aNWqoVatWSk5O1q+//urTc+/cuVMdOnRQSEiIbDabpk+f7vVz2Gw2TZo0yevH/TOpqamy2Wyy2Wz65JNPiqy3LEvXXHONbDabYmNjPTrHrFmzlJqaWqp9PvnkkwvWBMAzVSu6AKCk5s2bp2HDhum6667T6NGj1aRJE+Xn52vbtm2aM2eOtmzZopUrV/rs/I888ohycnK0bNky1apVS1dffbXXz7Flyxb95S9/8fpxS6pmzZqaP39+kXDfsGGDvvvuO9WsWdPjY8+aNUu1a9fWgAEDSrxPq1attGXLFjVp0sTj8wJwR/CjUtiyZYuGDh2qLl26aNWqVbLb7a51Xbp00ciRI7VmzRqf1rBr1y4NHjxYcXFxPjvH3/72N58duyT69OmjN954QzNnzlRwcLCrff78+WrXrp2OHz9eLnXk5+fLZrMpODi4wv9MgMsNQ/2oFBITE2Wz2TR37ly30D+nWrVq6tmzp+t3YWGhkpOTdf3118tut6tOnTp66KGH9NNPP7ntFxsbq2bNmikjI0Pt27dXYGCgGjZsqBdffFGFhYWS/m8Y/OzZs5o9e7ZrSFySJk2a5Pr3Pzq3z/79+11t69atU2xsrMLCwhQQEKD69evr7rvv1qlTp1zbFDfUv2vXLt15552qVauWqlevrpYtW2rRokVu25wbEn/zzTc1btw4RUREKDg4WJ07d9Y333xTsj9kSffff78k6c0333S1HTt2TO+8844eeeSRYvd57rnn1LZtW4WGhio4OFitWrXS/Pnz9cfvf1199dXavXu3NmzY4PrzOzdicq72xYsXa+TIkapXr57sdrv27dtXZKj/8OHDioyMVExMjPLz813H37Nnj4KCgvTggw+W+FoBUxH8uOQVFBRo3bp1at26tSIjI0u0z9ChQzV27Fh16dJFq1ev1vPPP681a9YoJiZGhw8fdtvW6XSqX79+euCBB7R69WrFxcUpISFBS5YskSR1795dW7ZskSTdc8892rJli+t3Se3fv1/du3dXtWrVtGDBAq1Zs0YvvviigoKCdObMmQvu98033ygmJka7d+/WK6+8ohUrVqhJkyYaMGCAkpOTi2z/7LPP6ocfftBrr72muXPn6ttvv1WPHj1UUFBQojqDg4N1zz33aMGCBa62N998U1WqVFGfPn0ueG1DhgzRW2+9pRUrVuiuu+7SE088oeeff961zcqVK9WwYUNFR0e7/vzOvy2TkJCgAwcOaM6cOXr33XdVp06dIueqXbu2li1bpoyMDI0dO1aSdOrUKd17772qX7++5syZU6LrBIxmAZc4p9NpSbLuu+++Em3/9ddfW5KsYcOGubV/9tlnliTr2WefdbV16NDBkmR99tlnbts2adLEuu2229zaJFnDhw93a5s4caJV3P+NFi5caEmysrKyLMuyrLffftuSZGVmZl60dknWxIkTXb/vu+8+y263WwcOHHDbLi4uzgoMDLR+++03y7Isa/369ZYkq1u3bm7bvfXWW5Yka8uWLRc977l6MzIyXMfatWuXZVmWdcMNN1gDBgywLMuymjZtanXo0OGCxykoKLDy8/OtyZMnW2FhYVZhYaFr3YX2PXe+W2655YLr1q9f79Y+ZcoUS5K1cuVKq3///lZAQID15ZdfXvQaAfyOHj8uO+vXr5ekIpPIbrzxRjVu3Fgff/yxW3t4eLhuvPFGt7a//vWv+uGHH7xWU8uWLVWtWjU9+uijWrRokb7//vsS7bdu3Tp16tSpyEjHgAEDdOrUqSIjD3+83SH9fh2SSnUtHTp0UKNGjbRgwQJ99dVXysjIuOAw/7kaO3furJCQEPn5+cnf318TJkzQkSNHlJ2dXeLz3n333SXedvTo0erevbvuv/9+LVq0SDNmzFDz5s1LvD9gMoIfl7zatWsrMDBQWVlZJdr+yJEjkqS6desWWRcREeFaf05YWFiR7ex2u3Jzcz2otniNGjXSRx99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",
      "text/plain": [
       "<Figure size 600x400 with 2 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.figure(figsize=(6,4))\n",
    "sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')\n",
    "plt.title(\"Confusion Matrix\")\n",
    "plt.xlabel(\"Predicted\")\n",
    "plt.ylabel(\"Actual\")\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "414a9cd0-1102-4cc9-862f-d68734bcd86c",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAA5AAAAGHCAYAAADGAL0uAAAAOnRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjEwLjAsIGh0dHBzOi8vbWF0cGxvdGxpYi5vcmcvlHJYcgAAAAlwSFlzAAAPYQAAD2EBqD+naQAAfzxJREFUeJzs3XlcTvn/P/7H1Xa1XxKSpNBOCYkylMFEeFuG7KmsY1+jIVP2dTB8LWMp21jGNsbYKbKFCKMsE00hspZCqev8/vDrfFzarlJK87jfbuc2zuv1Oq/zPOd0za1nr9d5XRJBEAQQERERERERFUKlrAMgIiIiIiKirwMTSCIiIiIiIlIKE0giIiIiIiJSChNIIiIiIiIiUgoTSCIiIiIiIlIKE0giIiIiIiJSChNIIiIiIiIiUgoTSCIiIiIiIlIKE0giIiIiIiJSChNIIiKiMiaRSJTawsPDSz2WTZs2oVevXrC2toaKigrMzc3zbZuWloaxY8eiRo0a0NTUhKOjI7Zv367UeYKCgiCRSPDs2bMSivzLW7lyJUJDQ7/Iuc6dO4egoCC8evVKqfY59zevbcWKFeUiRiL6OqmVdQBERET/defPn1fYnzlzJsLCwnDy5EmFcjs7u1KPZfPmzXj8+DGcnZ0hl8vx/v37fNt269YNly5dwrx582BlZYXffvsNvXv3hlwuR58+fUo91rK2cuVKVKlSBT4+PqV+rnPnziE4OBg+Pj6oVKmS0scdPnwYMplMoax27dolHN0HxY2RiL4uTCCJiIjKWLNmzRT2q1atChUVlVzlX8KRI0egovJhglLHjh3x999/59nu4MGDOHbsmJg0AkCrVq3w77//YtKkSejZsydUVVW/WNxf0ps3b6CtrV3WYSilcePGqFKlSlmH8Vnevn0LTU1NSCSSsg6FiMAprERERF+FFy9eYPjw4TAxMYGGhgbq1KmDqVOnIiMjQ6GdRCLByJEjsWbNGlhZWUEqlcLOzk7pqaU5yWNh9u7dC11dXfTo0UOh3NfXF48ePUJkZKRyF/YRd3d31K9fH+fPn4erqyu0tLRgbm6OkJAQAMBff/2FRo0aQVtbG/b29jh8+LDC8TnTNq9evYpu3bpBX18fMpkM/fr1w9OnTxXayuVyLFiwADY2NpBKpahWrRq8vb3x4MGDPGM6ffo0XF1doa2tDT8/P5ibm+PmzZs4deqUODU0Z7rvu3fvMGHCBDg6OkImk6Fy5cpwcXHBH3/8keuac57X5s2bYWtrC21tbTRo0AAHDhxQuK5JkyYB+DB6WFJTmgVBwMqVK+Ho6AgtLS0YGBige/fuuHfvnkK7Y8eOoXPnzqhZsyY0NTVhYWGBoUOHKkw/LixGiUSCoKCgXDGYm5srjOCGhoZCIpHg6NGj8PPzQ9WqVaGtrS3+nO/YsQMuLi7Q0dGBrq4uPDw8cPXqVYU+7927h169eqFGjRqQSqUwMjJC69atER0d/Vn3i4g+4AgkERFROffu3Tu0atUKcXFxCA4OhoODAyIiIjB37lxER0fjr7/+Umi/f/9+hIWFYcaMGdDR0cHKlSvRu3dvqKmpoXv37iUS099//w1bW1uoqSn+KuHg4CDWu7q6Frnfx48fw9fXF/7+/qhZsyaWL18OPz8/JCYmYteuXfjxxx8hk8kwY8YMdOnSBffu3UONGjUU+ujatSu8vLwwbNgw3Lx5E4GBgYiJiUFkZCTU1dUBAD/88AN+/fVXjBw5Eh07dkR8fDwCAwMRHh6OK1euKIzaJSUloV+/fvD398ecOXOgoqKCyZMno3v37pDJZFi5ciUAQCqVAgAyMjLw4sULTJw4ESYmJsjMzMTx48fRrVs3hISEwNvbWyHev/76C5cuXcKMGTOgq6uLBQsWoGvXrrh9+zbq1KmDQYMG4cWLF1i+fDn27NkDY2NjAMpNac7OzkZWVpa4L5FIxJHhoUOHIjQ0FKNHj8b8+fPx4sULzJgxA66urrh27RqMjIwAAHFxcXBxccGgQYMgk8kQHx+Pn3/+Gd988w1u3LgBdXX1z4oxL35+fujQoQM2b96M9PR0qKurY86cOZg2bRp8fX0xbdo0ZGZmYuHChWjRogUuXrwonsvT0xPZ2dlYsGABatWqhWfPnuHcuXN8N5OopAhERERUrgwYMEDQ0dER91evXi0AEHbu3KnQbv78+QIA4ejRo2IZAEFLS0t4/PixWJaVlSXY2NgIFhYWRYqjQ4cOgpmZWZ51lpaWgoeHR67yR48eCQCEOXPmFNj3Tz/9JAAQnj59Kpa5ubkJAITLly+LZc+fPxdUVVUFLS0t4eHDh2J5dHS0AED45ZdfcvU5btw4hXNt3bpVACBs2bJFEARBiI2NFQAIw4cPV2gXGRkpABB+/PHHXDGdOHEi1zXUq1dPcHNzK/A6BeHD/X///r0wcOBAoWHDhgp1AAQjIyMhNTVVLHv8+LGgoqIizJ07VyxbuHChAEC4f/9+oecThP+7F59uJiYmgiAIwvnz5wUAwuLFixWOS0xMFLS0tAR/f/88+5XL5cL79++Ff//9VwAg/PHHH0rFCED46aefcpWbmZkJAwYMEPdDQkIEAIK3t7dCu4SEBEFNTU0YNWqUQvnr16+F6tWrC15eXoIgCMKzZ88EAMLSpUvzvTdE9Hk4hZWIiKicO3nyJHR0dHKNHuZM/Ttx4oRCeevWrcXRIwBQVVVFz5498c8//+Saovk5CnonrbjvqxkbG6Nx48bifuXKlVGtWjU4OjoqjDTa2toCAP79999cffTt21dh38vLC2pqaggLCwMA8b+fLn7j7OwMW1vbXPfTwMAA3377bZGu4/fff0fz5s2hq6sLNTU1qKurY/369YiNjc3VtlWrVtDT0xP3jYyMUK1atTyvraiOHz+OS5cuidvBgwcBAAcOHIBEIkG/fv2QlZUlbtWrV0eDBg0UpscmJydj2LBhMDU1Fa/FzMwMAPK8npLw/fffK+wfOXIEWVlZ8Pb2VohXU1MTbm5uYryVK1dG3bp1sXDhQvz888+4evUq5HJ5qcRI9F/FKaxERETl3PPnz1G9evVcSVm1atWgpqaG58+fK5RXr149Vx85Zc+fP0fNmjU/OyZDQ8Nc5wU+vKsJfPhFvjjyOk5DQyNXuYaGBoAP03s/9en1q6mpKcSb89+caZYfq1GjRq7ELa92BdmzZw+8vLzQo0cPTJo0CdWrV4eamhpWrVqFDRs25GpvaGiYq0wqleLt27dFOm9eGjRokOciOk+ePIEgCAp/aPhYnTp1AHx4V/S7777Do0ePEBgYCHt7e+jo6EAul6NZs2YlEmNePr3nT548AQA0adIkz/Y57+5KJBKcOHECM2bMwIIFCzBhwgRUrlwZffv2xezZsxUSdSIqHiaQRERE5ZyhoSEiIyMhCIJCEpmcnIysrKxcCcLjx49z9ZFTlleyUhz29vbYtm0bsrKyFN6DvHHjBgCgfv36JXKe4nj8+DFMTEzE/aysLDx//ly89pz/JiUl5UqmHz16lOt+FnU0dcuWLahduzZ27NihcOynCx6VpSpVqkAikSAiIkJ8d/NjOWV///03rl27htDQUAwYMECs/+eff4p0PqlUmuf15/VHCCD3Pc95Jrt27RJHP/NjZmaG9evXAwDu3LmDnTt3IigoCJmZmVi9enWR4iai3DiFlYiIqJxr3bo10tLSsG/fPoXyTZs2ifUfO3HihDhiA3xYSGXHjh2oW7duiYw+Ah8WqklLS8Pu3bsVyjdu3IgaNWqgadOmJXKe4ti6davC/s6dO5GVlQV3d3cAEKejbtmyRaHdpUuXEBsbm+t+5ie/UUKJRAINDQ2FJOjx48d5rsKqrJyErqRG/Dp27AhBEPDw4UM4OTnl2uzt7QH8XyL3aZK5Zs2aIsVobm6O69evK5SdPHkSaWlpSsXr4eEBNTU1xMXF5Rmvk5NTnsdZWVlh2rRpsLe3x5UrV5Q6FxEVjCOQRERE5Zy3tzf+3//7fxgwYADi4+Nhb2+PM2fOYM6cOfD09ESbNm0U2lepUgXffvstAgMDxVVYb926pdRXecTExCAmJgbAh6TnzZs32LVrF4APK2rmrHTZvn17tG3bFj/88ANSU1NhYWGBbdu24fDhw9iyZUuZfgfknj17oKamhrZt24qrsDZo0ABeXl4AAGtrawwZMgTLly+HiooK2rdvL67CampqinHjxil1Hnt7e2zfvh07duxAnTp1oKmpCXt7e3Ts2BF79uzB8OHD0b17dyQmJmLmzJkwNjbG3bt3i3VNOQndsmXLMGDAAKirq8Pa2rrYUzKbN2+OIUOGwNfXF5cvX0bLli2ho6ODpKQknDlzBvb29vjhhx9gY2ODunXrYsqUKRAEAZUrV8aff/6JY8eOFSnG/v37IzAwENOnT4ebmxtiYmKwYsUKyGQypeI1NzfHjBkzMHXqVNy7dw/t2rWDgYEBnjx5gosXL0JHRwfBwcG4fv06Ro4ciR49esDS0hIaGho4efIkrl+/jilTphTrXhHRJ8p2DR8iIiL61KersArCh9VIhw0bJhgbGwtqamqCmZmZEBAQILx7906hHQBhxIgRwsqVK4W6desK6urqgo2NjbB161alzp3f6p3IYxXN169fC6NHjxaqV68uaGhoCA4ODsK2bduKdJ5PV2GtV69errZmZmZChw4dcpXnXOunfUZFRQmdOnUSdHV1BT09PaF3797CkydPFI7Nzs4W5s+fL1hZWQnq6upClSpVhH79+gmJiYkK7fKLSRAEIT4+Xvjuu+8EPT09AYDCirXz5s0TzM3NBalUKtja2gpr164V4yvoGj6+5o9XJxUEQQgICBBq1KghqKioCACEsLCwPOP6+F58fH/zsmHDBqFp06aCjo6OoKWlJdStW1fw9vZWWAk3JiZGaNu2raCnpycYGBgIPXr0EBISEvL8mcgvxoyMDMHf318wNTUVtLS0BDc3NyE6OjrfVVgvXbqUZ7z79u0TWrVqJejr6wtSqVQwMzMTunfvLhw/flwQBEF48uSJ4OPjI9jY2Ag6OjqCrq6u4ODgICxZskTIysoq8F4QkXIkgiAIXzppJSIiotIhkUgwYsQIrFixoqxD+eKCgoIQHByMp0+f5rlwDBERfT6+A0lERERERERKYQJJRERERERESuEUViIiIiIiIlIKRyCJiIiIiIhIKUwgiYiIiIiISClMIImIiIiIiEgpamUdABFVbHK5HI8ePYKenh4kEklZh0NEREREnxAEAa9fv0aNGjWgolLwGCMTSCIqVY8ePYKpqWlZh0FEREREhUhMTETNmjULbMMEkohKlZ6eHoAP/0PS19cv42iIiIiI6FOpqakwNTUVf28rCBNIIipVOdNW9fX1mUASERERlWPKvG7EBJKIvoiW07ZBVapV1mEQERERlXtRC73LOoR8cRVWIiIiIiIiUgoTSCIiIiIiIlIKE0giIiIiIiJSChNIIiIiIiIiUgoTSCIiIiIiIlIKE0giIiIiIiJSChNI+k8KCgqCo6NjgW3c3d0xduzYLxKPRCLBvn37vsi5iIiIiIiKiwkkVXhMzoiIiIiISgYTSKIKKjMzs6xDICIiIqIKhgkkfTHu7u4YNWoUxo4dCwMDAxgZGeHXX39Feno6fH19oaenh7p16+LQoUPiMadOnYKzszOkUimMjY0xZcoUZGVlKfQ5evRo+Pv7o3LlyqhevTqCgoLEenNzcwBA165dIZFIxP0cmzdvhrm5OWQyGXr16oXXr1/nGfuMGTNgb2+fq7xx48aYPn26Ute/YcMG1KtXT7yWkSNHKtQ/e/YMXbt2hba2NiwtLbF//36xLjs7GwMHDkTt2rWhpaUFa2trLFu2TOF4Hx8fdOnSBXPnzkWNGjVgZWUFADh37hwcHR2hqakJJycn7Nu3DxKJBNHR0eKxMTEx8PT0hK6uLoyMjNC/f388e/ZMrN+1axfs7e2hpaUFQ0NDtGnTBunp6UpdNxERERFVHEwg6YvauHEjqlSpgosXL2LUqFH44Ycf0KNHD7i6uuLKlSvw8PBA//798ebNGzx8+BCenp5o0qQJrl27hlWrVmH9+vWYNWtWrj51dHQQGRmJBQsWYMaMGTh27BgA4NKlSwCAkJAQJCUlifsAEBcXh3379uHAgQM4cOAATp06hXnz5uUZt5+fH2JiYhSOv379Oq5evQofH59Cr3vVqlUYMWIEhgwZghs3bmD//v2wsLBQaBMcHAwvLy9cv34dnp6e6Nu3L168eAEAkMvlqFmzJnbu3ImYmBhMnz4dP/74I3bu3KnQx4kTJxAbG4tjx47hwIEDeP36NTp16gR7e3tcuXIFM2fOxOTJkxWOSUpKgpubGxwdHXH58mUcPnwYT548gZeXl1jfu3dv+Pn5ITY2FuHh4ejWrRsEQcjzWjMyMpCamqqwEREREVHFIBHy+y2QqIS5u7sjOzsbERERAD6MqslkMnTr1g2bNm0CADx+/BjGxsY4f/48/vzzT+zevRuxsbGQSCQAgJUrV2Ly5MlISUmBiopKrj4BwNnZGd9++62YDEokEuzduxddunQR2wQFBWHhwoV4/Pgx9PT0AAD+/v44ffo0Lly4IMbr6OiIpUuXAgA8PT1hbm6OlStXAgDGjRuH6OhohIWFFXrtJiYm8PX1zZX85pBIJJg2bRpmzpwJAEhPT4eenh4OHjyIdu3a5XnMiBEj8OTJE+zatQvAhxHIw4cPIyEhARoaGgCA1atXY9q0aXjw4AE0NTUBAOvWrcPgwYNx9epVODo6Yvr06YiMjMSRI0fEvh88eABTU1Pcvn0baWlpaNy4MeLj42FmZlbotQYFBSE4ODhXeYNRq6Eq1Sr0eCIiIqL/uqiF3l/0fKmpqZDJZEhJSYG+vn6BbTkCSV+Ug4OD+G9VVVUYGhoqTA01MjICACQnJyM2NhYuLi5i8ggAzZs3R1paGh48eJBnnwBgbGyM5OTkQmMxNzcXk0dljhs8eDC2bduGd+/e4f3799i6dSv8/PwKPU9ycjIePXqE1q1bF9ju4+vQ0dGBnp6eQjyrV6+Gk5MTqlatCl1dXaxduxYJCQkKfdjb24vJIwDcvn0bDg4OYvIIfEiwPxYVFYWwsDDo6uqKm42NDYAPo7QNGjRA69atYW9vjx49emDt2rV4+fJlvtcREBCAlJQUcUtMTCzwuomIiIjo66FW1gHQf4u6urrCvkQiUSjLSRblcjkEQVBIHgGI0yY/Ls+rT7lcXqxYCjquU6dOkEql2Lt3L6RSKTIyMvD9998Xeh4tLeVG3QqKZ+fOnRg3bhwWL14MFxcX6OnpYeHChYiMjFQ4RkdHR2G/oHuYQy6Xo1OnTpg/f36umIyNjaGqqopjx47h3LlzOHr0KJYvX46pU6ciMjIStWvXznWMVCqFVCpV6pqJiIiI6OvCEUgqt+zs7HDu3DmFhOfcuXPQ09ODiYmJ0v2oq6sjOzv7s+NRU1PDgAEDEBISgpCQEPTq1Qva2tqFHqenpwdzc3OcOHGi2OeOiIiAq6srhg8fjoYNG8LCwgJxcXGFHmdjY4Pr168jIyNDLLt8+bJCm0aNGuHmzZswNzeHhYWFwpaTkEokEjRv3hzBwcG4evUqNDQ0sHfv3mJfDxERERF9nZhAUrk1fPhwJCYmYtSoUbh16xb++OMP/PTTTxg/fjxUVJT/0c1J3h4/flzg1EtlDBo0CCdPnsShQ4eUmr6aIygoCIsXL8Yvv/yCu3fv4sqVK1i+fLnSx1tYWODy5cs4cuQI7ty5g8DAQIUFffLTp08fyOVyDBkyBLGxsThy5AgWLVoE4P9GcUeMGIEXL16gd+/euHjxIu7du4ejR4/Cz88P2dnZiIyMxJw5c3D58mUkJCRgz549ePr0KWxtbZWOn4iIiIgqBiaQVG6ZmJjg4MGDuHjxIho0aIBhw4Zh4MCBmDZtWpH6Wbx4MY4dOwZTU1M0bNjws2KytLSEq6srrK2t0bRpU6WPGzBgAJYuXYqVK1eiXr166NixI+7evav08cOGDUO3bt3Qs2dPNG3aFM+fP8fw4cMLPU5fXx9//vknoqOj4ejoiKlTp4pfO5LzXmSNGjVw9uxZZGdnw8PDA/Xr18eYMWMgk8mgoqICfX19nD59Gp6enrCyssK0adOwePFitG/fXun4iYiIiKhi4CqsREUgCAJsbGwwdOhQjB8/vqzDKZatW7fC19cXKSkpSr+f+TlyVvXiKqxEREREyinPq7ByER0iJSUnJ2Pz5s14+PAhfH19yzocpW3atAl16tSBiYkJrl27hsmTJ8PLy+uLJI9EREREVLEwgSRSkpGREapUqYJff/0VBgYGCnW6urr5Hnfo0CG0aNGitMPL1+PHjzF9+nTxOzZ79OiB2bNnl1k8RERERPT1YgJJpKSCZntHR0fnW1eUFWNLg7+/P/z9/cs0BiIiIiKqGJhAEpUACwuLsg6BiIiIiKjUcRVWIiIiIiIiUgpHIInoizg9q3ehq3oRERERUfnGEUgiIiIiIiJSChNIIiIiIiIiUgoTSCIiIiIiIlIKE0giIiIiIiJSChNIIiIiIiIiUgoTSCIiIiIiIlIKv8aDiL6IltO2QVWqVdZhEBERUSmLWuhd1iFQKeIIJBERERERESmFCSQREREREREphQkkERERERERKYUJJBERERERESmFCSQREREREREphQkkERERERERKYUJJBERERERESmFCSRROePu7o6xY8eWdRhERERERLkwgSSiPAmCgKysrLIOg4iIiIjKESaQROWIj48PTp06hWXLlkEikUAikSA+Ph4xMTHw9PSErq4ujIyM0L9/fzx79kw8zt3dHaNHj4a/vz8qV66M6tWrIygoSKyPj4+HRCJBdHS0WPbq1StIJBKEh4cDAMLDwyGRSHDkyBE4OTlBKpUiIiICgiBgwYIFqFOnDrS0tNCgQQPs2rXrC90RIiIiIipPmEASlSPLli2Di4sLBg8ejKSkJCQlJUFdXR1ubm5wdHTE5cuXcfjwYTx58gReXl4Kx27cuBE6OjqIjIzEggULMGPGDBw7dqzIMfj7+2Pu3LmIjY2Fg4MDpk2bhpCQEKxatQo3b97EuHHj0K9fP5w6dSrP4zMyMpCamqqwEREREVHFoFbWARDR/5HJZNDQ0IC2tjaqV68OAJg+fToaNWqEOXPmiO02bNgAU1NT3LlzB1ZWVgAABwcH/PTTTwAAS0tLrFixAidOnEDbtm2LFMOMGTPEY9LT0/Hzzz/j5MmTcHFxAQDUqVMHZ86cwZo1a+Dm5pbr+Llz5yI4OLjoF09ERERE5R4TSKJyLioqCmFhYdDV1c1VFxcXp5BAfszY2BjJyclFPp+Tk5P475iYGLx79y5XEpqZmYmGDRvmeXxAQADGjx8v7qempsLU1LTIcRARERFR+cMEkqick8vl6NSpE+bPn5+rztjYWPy3urq6Qp1EIoFcLgcAqKh8mK0uCIJY//79+zzPp6Ojo3BuAPjrr79gYmKi0E4qleZ5vFQqzbeOiIiIiL5uTCCJyhkNDQ1kZ2eL+40aNcLu3bthbm4ONbXifWSrVq0KAEhKShJHDj9eUCc/dnZ2kEqlSEhIyHO6KhERERH9tzCBJCpnzM3NERkZifj4eOjq6mLEiBFYu3YtevfujUmTJqFKlSr4559/sH37dqxduxaqqqqF9qmlpYVmzZph3rx5MDc3x7NnzzBt2rRCj9PT08PEiRMxbtw4yOVyfPPNN0hNTcW5c+egq6uLAQMGlMQlExEREdFXgquwEpUzEydOhKqqKuzs7FC1alVkZmbi7NmzyM7OhoeHB+rXr48xY8ZAJpOJU1OVsWHDBrx//x5OTk4YM2YMZs2apdRxM2fOxPTp0zF37lzY2trCw8MDf/75J2rXrl3cSyQiIiKir5RE+PilKCKiEpaamgqZTIYGo1ZDVapV1uEQERFRKYta6F3WIVAR5fy+lpKSAn19/QLbcgSSiIiIiIiIlMIEkoiIiIiIiJTCBJKIiIiIiIiUwgSSiIiIiIiIlMIEkoiIiIiIiJTC74Ekoi/i9Kzeha7qRURERETlG0cgiYiIiIiISClMIImIiIiIiEgpTCCJiIiIiIhIKUwgiYiIiIiISClMIImIiIiIiEgpXIWViL6IltO2QVWqVdZhEBERVWhRC73LOgSq4DgCSUREREREREphAklERERERERKYQJJRERERERESmECSUREREREREphAklERERERERKYQJJRERERERESmECSUREREREREphAllC3N3dMXbs2LIO4z8hPDwcEokEr169KldxhIaGolKlSmUaU0mRSCTYt29fWYdBREREROUME8gSsmfPHsycOVPp9vHx8ZBIJIiOji69oIroa0mAXF1dkZSUBJlMVurnynlOn279+vX7onGUlqCgIDg6OuYqT0pKQvv27b98QERERERUrqmVdQAVReXKlcvs3O/fv4e6unqZnb+kCIKA7OxsqKkV/GOpoaGB6tWrf6GoPjh+/Djq1asn7mtpaX2xOMri+X7p+0tEREREXweOQJaQT6ewmpubY86cOfDz84Oenh5q1aqFX3/9VayvXbs2AKBhw4aQSCRwd3cX60JCQmBrawtNTU3Y2Nhg5cqVYl3OiNjOnTvh7u4OTU1NbNmyBT4+PujSpQsWLVoEY2NjGBoaYsSIEXj//r14bGZmJvz9/WFiYgIdHR00bdoU4eHhAD5Mx/T19UVKSoo4yhYUFFToda9cuRKWlpbQ1NSEkZERunfvLtYJgoAFCxagTp060NLSQoMGDbBr1y6xPmcK6JEjR+Dk5ASpVIr169dD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Jk2awNDQEGfPnsXChQvF73gkIiIiIioKJpBEnykhIQF2dnb51sfExKBWrVpfMCJFd+/exaxZs/DixQvUqlULEyZMQEBAQJnFQ0RERERfLyaQRJ+pRo0aiI6OLrC+LC1ZsgRLliwp0xiIiIiIqGJgAkn0mdTU1Ar8CgwiIiIiooqCq7ASERERERGRUjgCSURfxOlZvQtd1YuIiIiIyjeOQBIREREREZFSmEASERERERGRUphAEhERERERkVKYQBIREREREZFSmEASERERERGRUrgKKxF9ES2nbYOqVKusw6ASFrXQu6xDICIioi+II5BERERERESkFCaQREREREREpBQmkERERERERKQUJpBERERERESkFCaQREREREREpJRiJ5CbN29G8+bNUaNGDfz7778AgKVLl+KPP/4oseCIiIiIiIio/ChWArlq1SqMHz8enp6eePXqFbKzswEAlSpVwtKlS0syPiIiIiIiIionipVALl++HGvXrsXUqVOhqqoqljs5OeHGjRslFhwRERERERGVH8VKIO/fv4+GDRvmKpdKpUhPT//soIiIiIiIiKj8KVYCWbt2bURHR+cqP3ToEOzs7D43JiIqQYcPH8Y333yDSpUqwdDQEB07dkRcXJxYf+7cOTg6OkJTUxNOTk7Yt28fJBKJwmc8JiYGnp6e0NXVhZGREfr3749nz56VwdUQERERUVkqVgI5adIkjBgxAjt27IAgCLh48SJmz56NH3/8EZMmTSrpGInoM6Snp2P8+PG4dOkSTpw4ARUVFXTt2hVyuRyvX79Gp06dYG9vjytXrmDmzJmYPHmywvFJSUlwc3ODo6MjLl++jMOHD+PJkyfw8vLK83wZGRlITU1V2IiIiIioYlArzkG+vr7IysqCv78/3rx5gz59+sDExATLli1Dr169SjpGIvoM33//vcL++vXrUa1aNcTExODMmTOQSCRYu3YtNDU1YWdnh4cPH2Lw4MFi+1WrVqFRo0aYM2eOWLZhwwaYmprizp07sLKyUuh/7ty5CA4OLt2LIiIiIqIyUeQRyKysLGzcuBGdOnXCv//+i+TkZDx+/BiJiYkYOHBgacRIRJ8hLi4Offr0QZ06daCvr4/atWsDABISEnD79m04ODhAU1NTbO/s7KxwfFRUFMLCwqCrqytuNjY2Yt+fCggIQEpKirglJiaW4tURERER0ZdU5BFINTU1/PDDD4iNjQUAVKlSpcSDIqKS06lTJ5iammLt2rWoUaMG5HI56tevj8zMTAiCAIlEotBeEASFfblcjk6dOmH+/Pm5+jY2Ns5VJpVKIZVKS/YiiIiIiKhcKNYU1qZNm+Lq1aswMzMr6XiIqAQ9f/4csbGxWLNmDVq0aAEAOHPmjFhvY2ODrVu3IiMjQ0z6Ll++rNBHo0aNsHv3bpibm0NNrVj/yyAiIiKiCqJYi+gMHz4cEyZMwIoVK3D+/Hlcv35dYSOi8sHAwACGhob49ddf8c8//+DkyZMYP368WN+nTx/I5XIMGTIEsbGxOHLkCBYtWgQA4sjkiBEj8OLFC/Tu3RsXL17EvXv3cPToUfj5+SE7O7tMrouIiIiIykaxhhN69uwJABg9erRYJpFIxOlw/KWSqHxQUVHB9u3bMXr0aNSvXx/W1tb45Zdf4O7uDgDQ19fHn3/+iR9++AGOjo6wt7fH9OnT0adPH/G9yBo1auDs2bOYPHkyPDw8kJGRATMzM7Rr1w4qKsX6GxQRERERfaWKlUDev3+/pOMgolLSpk0bxMTEKJR9/J6jq6srrl27Ju5v3boV6urqqFWrllhmaWmJPXv2lH6wRERERFSuFSuB5LuPRBXHpk2bUKdOHZiYmODatWuYPHkyvLy8oKWlVdahEREREVE5U6wEctOmTQXWe3t7FysYIvryHj9+jOnTp+Px48cwNjZGjx49MHv27LIOi4iIiIjKIYnw6Zr9SjAwMFDYf//+Pd68eQMNDQ1oa2vjxYsXJRYgEX3dUlNTIZPJ0GDUaqhKOapZ0UQt5B8MiYiIvnY5v6+lpKRAX1+/wLbFWgHj5cuXCltaWhpu376Nb775Btu2bStW0ERERERERFS+ldgSipaWlpg3bx7GjBlTUl0SERERERFROVKia/Crqqri0aNHJdklERERERERlRPFWkRn//79CvuCICApKQkrVqxA8+bNSyQwIqpYTs/qXeiceiIiIiIq34qVQHbp0kVhXyKRoGrVqvj222+xePHikoiLiIiIiIiIypliJZByubyk4yAiIiIiIqJyrljvQM6YMQNv3rzJVf727VvMmDHjs4MiIiIiIiKi8qdY3wOpqqqKpKQkVKtWTaH8+fPnqFatGrKzs0ssQCL6uhXle4WIiIiI6Msr9e+BFAQBEokkV/m1a9dQuXLl4nRJRERERERE5VyR3oE0MDCARCKBRCKBlZWVQhKZnZ2NtLQ0DBs2rMSDJKKvX8tp26Aq1SrrMCqMqIXeZR0CERER/QcVKYFcunQpBEGAn58fgoODIZPJxDoNDQ2Ym5vDxcWlxIMkIiIiIiKislekBHLAgAEAgNq1a8PV1RXq6uqlEhQRERERERGVP8X6Gg83Nzfx32/fvsX79+8V6rlQBhERERERUcVTrEV03rx5g5EjR6JatWrQ1dWFgYGBwkZEREREREQVT7ESyEmTJuHkyZNYuXIlpFIp1q1bh+DgYNSoUQObNm0q6RiJiIiIiIioHCjWFNY///wTmzZtgru7O/z8/NCiRQtYWFjAzMwMW7duRd++fUs6TiIiIiIiIipjxRqBfPHiBWrXrg3gw/uOL168AAB88803OH36dMlFR0REREREROVGsRLIOnXqID4+HgBgZ2eHnTt3AvgwMlmpUqWSio1IZG5ujqVLl36Rc4WGhhbr5/jmzZvw8vJC1apVIZVKYWlpicDAQLx586bkgyQiIiIiKgPFSiB9fX1x7do1AEBAQID4LuS4ceMwadKkEg2QSFnZ2dmQy+Vlcu4LFy6gadOmyMzMxF9//YU7d+5gzpw52LhxI9q2bYvMzMwyiau0fboCMxERERFVbMVKIMeNG4fRo0cDAFq1aoVbt25h27ZtuHLlCsaMGVOiAVL5J5fLMX/+fFhYWEAqlaJWrVqYPXu2WH/jxg18++230NLSgqGhIYYMGYK0tDSx3sfHB126dMGiRYtgbGwMQ0NDjBgxQkxO3N3d8e+//2LcuHGQSCSQSCQA/m+k8MCBA7Czs4NUKsW///6LS5cuoW3btqhSpQpkMhnc3Nxw5coVhZhfvXqFIUOGwMjICJqamqhfvz4OHDiA8PBw+Pr6IiUlRTxXUFBQgdcvCAIGDhwIW1tb7NmzB87OzjAzM0OPHj3w559/4vz581iyZInYXiKRYN26dejatSu0tbVhaWmJ/fv3K/QZExMDT09P6OrqwsjICP3798ezZ8/yPH96ejr09fWxa9cuhfI///wTOjo6eP36NQDg4cOH6NmzJwwMDGBoaIjOnTuLMwkAKHXfJBIJVq9ejc6dO0NHRwezZs0q8N4QERERUcVSrATyY+/evUOtWrXQrVs3NGjQoCRioq9MQEAA5s+fj8DAQMTExOC3336DkZERgA9f+dKuXTsYGBjg0qVL+P3333H8+HGMHDlSoY+wsDDExcUhLCwMGzduRGhoKEJDQwEAe/bsQc2aNTFjxgwkJSUhKSlJPO7NmzeYO3cu1q1bh5s3b6JatWp4/fo1BgwYgIiICFy4cAGWlpbw9PQUEym5XI727dvj3Llz2LJlC2JiYjBv3jyoqqrC1dUVS5cuhb6+vniuiRMnFnj90dHRiImJwfjx46GioviRatCgAdq0aYNt27YplAcHB8PLywvXr1+Hp6cn+vbtK75LnJSUBDc3Nzg6OuLy5cs4fPgwnjx5Ai8vrzzPr6Ojg169eiEkJEShPCQkBN27d4eenh7evHmDVq1aQVdXF6dPn8aZM2egq6uLdu3aiaOjhd23HD/99BM6d+6MGzduwM/PL1c8GRkZSE1NVdiIiIiIqGIo1iqs2dnZmDNnDlavXo0nT57gzp07qFOnDgIDA2Fubo6BAweWdJxUTr1+/RrLli3DihUrMGDAAABA3bp18c033wAAtm7dirdv32LTpk3Q0dEBAKxYsQKdOnXC/PnzxUTTwMAAK1asgKqqKmxsbNChQwecOHECgwcPRuXKlaGqqgo9PT1Ur15d4fzv37/HypUrFf548e233yq0WbNmDQwMDHDq1Cl07NgRx48fx8WLFxEbGwsrKysAH97rzSGTySCRSHKdKz937twBANja2uZZb2trizNnziiU+fj4oHfv3gCAOXPmYPny5bh48SLatWuHVatWoVGjRpgzZ47YfsOGDTA1NcWdO3fEmD82aNAguLq64tGjR6hRowaePXuGAwcO4NixYwCA7du3Q0VFBevWrRNHcENCQlCpUiWEh4fju+++K/S+5ejTp0+eiWOOuXPnIjg4ON96IiIiIvp6FWsEcvbs2QgNDcWCBQugoaEhltvb22PdunUlFhyVf7GxscjIyEDr1q3zrW/QoIGYPAJA8+bNIZfLcfv2bbGsXr16UFVVFfeNjY2RnJxc6Pk1NDTg4OCgUJacnIxhw4bBysoKMpkMMpkMaWlpSEhIAPBhxLBmzZp5JmKlQRAEMWnL8XHMOjo60NPTE683KioKYWFh0NXVFTcbGxsAQFxcXJ7ncHZ2Rr169cTvYd28eTNq1aqFli1bin3+888/0NPTE/usXLky3r17J/ZZ2H3L4eTkVOD1BgQEICUlRdwSExOVvVVEREREVM4VawRy06ZN+PXXX9G6dWsMGzZMLHdwcMCtW7dKLDgq/7S0tAqszyt5yvFxubq6eq46ZRbE0dLSytW/j48Pnj59iqVLl8LMzAxSqRQuLi7iVM3CYi6qnEQ0JiYGjo6Ouepv3boFS0tLhbKCrlcul4sjtJ8yNjbON45BgwZhxYoVmDJlCkJCQuDr6yveG7lcjsaNG2Pr1q25jqtatSqAwu9bjo//GJAXqVQKqVRaYBsiIiIi+joVawTy4cOHsLCwyFUul8u5KuN/jKWlJbS0tHDixIk86+3s7BAdHY309HSx7OzZs1BRUSnSCKCGhgays7OVahsREYHRo0fD09MT9erVg1QqVViAxsHBAQ8ePBCnnn7OuQDA0dERNjY2WLJkSa6k99q1azh+/Lg4XVUZjRo1ws2bN2Fubg4LCwuFraDkrV+/fkhISMAvv/yCmzdvilOKc/q8e/cuqlWrlqtPmUwGoPD7RkRERERUrASyXr16iIiIyFX++++/o2HDhp8dFH09NDU1MXnyZPj7+2PTpk2Ii4vDhQsXsH79egBA3759oampiQEDBuDvv/9GWFgYRo0ahf79+4vvPyrD3Nwcp0+fxsOHDwtNaiwsLLB582bExsYiMjISffv2VRh1dHNzQ8uWLfH999/j2LFjuH//Pg4dOoTDhw+L50pLS8OJEyfw7NmzQr/HMWdV1ZiYGHz//fe4ePEiEhIS8Pvvv6NTp05wcXHB2LFjlb7WESNG4MWLF+jduzcuXryIe/fu4ejRo/Dz8yswsTUwMEC3bt0wadIkfPfdd6hZs6ZY17dvX1SpUgWdO3dGREQE7t+/j1OnTmHMmDF48OCBUveNiIiIiKhYCeRPP/2EkSNHYv78+ZDL5dizZw8GDx6MOXPmYPr06SUdI5VzgYGBmDBhAqZPnw5bW1v07NlTfJ9PW1sbR44cwYsXL9CkSRN0794drVu3xooVK4p0jhkzZiA+Ph5169YVp1zmZ8OGDXj58iUaNmyI/v37Y/To0ahWrZpCm927d6NJkybo3bs37Ozs4O/vLyZnrq6uGDZsGHr27ImqVatiwYIFhcbXvHlzXLhwAaqqqvD09ISFhQUCAgIwYMAAHDt2rEhTOmvUqIGzZ88iOzsbHh4eqF+/PsaMGQOZTJZrlddPDRw4EJmZmbkWudHW1sbp06fFFZNtbW3h5+eHt2/fQl9fX+n7RkRERET/bRJBEARlG9+7dw+1a9eGRCLBkSNHMGfOHERFRUEul6NRo0aYPn06vvvuu9KMl4gKsHXrVowZMwaPHj1SWOCqLKWmpkImk6HBqNVQlXJEs6RELfQu6xCIiIiogsj5fS0lJUUcXMhPkRbRsbS0RFJSEqpVqwYPDw9s2LAB//zzj9Jfd0BEpePNmze4f/8+5s6di6FDh5ab5JGIiIiIKpYiTWH9dLDy0KFDhb4fRvS1i4iIUPhKjU+38mDBggVwdHSEkZERAgICyjocIiIiIqqgivU1HjmKMPuV6Kvl5OSE6Ojosg6jQEFBQQgKCirrMIiIiIiogitSAimRSHJ9515+3/FHVFFoaWnl+bU1RERERET/NUVKIAVBgI+Pj7ii5Lt37zBs2LBc3023Z8+ekouQiIiIiIiIyoUiJZAffzE58OGLy4mIlHF6Vu9CV/UiIiIiovKtSAlkSEhIacVBRERERERE5VyRVmElIiIiIiKi/y4mkERERERERKQUJpBERERERESkFCaQREREREREpBQmkERERERERKSUIq3CSkRUXC2nbYOqVKuswyhzUQu9yzoEIiIiomLjCCQREREREREphQkkERERERERKYUJJBERERERESmFCSQREREREREphQkkERERERERKYUJJBERERERESmFCSQREREREREphQkkfVXMzc2xdOnSL3Ku0NBQVKpUSam28fHxkEgkBW5BQUGlGi8RERERUWlTK+sAiEpadnY2JBIJVFS+3N9HTE1NkZSUJO4vWrQIhw8fxvHjx8UyXV3dLxbPl5KZmQkNDY2yDoOIiIiIvhCOQFKJkcvlmD9/PiwsLCCVSlGrVi3Mnj1brL9x4wa+/fZbaGlpwdDQEEOGDEFaWppY7+Pjgy5dumDRokUwNjaGoaEhRowYgffv3wMA3N3d8e+//2LcuHHiqB7wfyOFBw4cgJ2dHaRSKf79919cunQJbdu2RZUqVSCTyeDm5oYrV64oxPzq1SsMGTIERkZG0NTURP369XHgwAGEh4fD19cXKSkpSo0gqqqqonr16uKmq6sLNTU1VK9eHVpaWjAxMcHt27cBAIIgoHLlymjSpIl4/LZt22BsbKz0vfqYIAiwsLDAokWLFMr//vtvqKioIC4uDgCQkpKCIUOGoFq1atDX18e3336La9euie3j4uLQuXNnGBkZQVdX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TERgYCC8vLzKTfJIRERERBULp7ASKSkiIkLhKzA+3crStm3bYG1tjZSUlM+eaktERERElB+OQBIpycnJCdHR0WUdRp58fHzg4+NT1mEQERERUQXHBJJISVpaWrCwsCjrMIiIiIiIygynsBIREREREZFSOAJJRF/E6Vm9C13Vi4iIiIjKN45AEhERERERkVKYQBIREREREZFSmEASERERERGRUphAEhERERERkVKYQBIREREREZFSuAorEX0RLadtg6pUq6zD+KKiFnqXdQhEREREJYojkERERERERKQUJpBERERERESkFCaQREREREREpBQmkERERERERKQUJpBERERERESkFCaQREREREREpBQmkERERERERKQUJpD/YeHh4ZBIJHj16lWJ9y2RSLBv374S6Ss0NBSVKlUqkb7KQnx8PCQSCaKjo0ulfx8fH3Tp0qVU+s7h7u6OsWPHluo5iIiIiKj8YwJZikozQSuq8pwASCSSPLft27eXyvm+dEJqamqKpKQk1K9f/4udk4iIiIioNKiVdQBEABASEoJ27doplJX1qGNmZiY0NDQ+ux9VVVVUr169BCIiIiIiIipbZToC6e7ujpEjR2LkyJGoVKkSDA0NMW3aNAiCILbZsmULnJycoKenh+rVq6NPnz5ITk4GAAiCAAsLCyxatEih37///hsqKiqIi4sD8GGEa82aNejYsSO0tbVha2uL8+fP459//oG7uzt0dHTg4uIits/x559/onHjxtDU1ESdOnUQHByMrKwssV4ikWDdunXo2rUrtLW1YWlpif379wP4MG2xVatWAAADAwNIJBL4+PgodU9GjRqFsWPHwsDAAEZGRvj111+Rnp4OX19f6OnpoW7dujh06JDCcTExMfD09ISuri6MjIzQv39/PHv2DMCHKY6nTp3CsmXLxNG9+Ph48dioqCg4OTlBW1sbrq6uuH37tkLfq1atQt26daGhoQFra2ts3rxZof7u3bto2bIlNDU1YWdnh2PHjhV6nZ+qVKkSqlevrrBpamrm276wZ/Pq1SsMGTIERkZG0NTURP369XHgwAGEh4fD19cXKSkp4r0ICgoCAJibm2PWrFnw8fGBTCbD4MGDAQC7d+9GvXr1IJVKYW5ujsWLFyvEYm5ujjlz5sDPzw96enqoVasWfv31V7E+rymsN2/eRIcOHaCvrw89PT20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4pKr/dGjR+Hk5AR1dfUC2+TXJxWstJ7TpwRBQHR0NIyNjUsm8P+g4jwrZfAzVbJK6zl9ip+pz1fcZ7Vt2zb4+Pjgt99+Q4cOHXLV8zNVskrrOX2Kn6lSUhYr9xDR12v79u2Curq6sH79eiEmJkYYO3asoKOjI8THxwuCIAhTpkwR+vfvL7a/d++eoK2tLYwbN06IiYkR1q9fL6irqwu7du0S25w9e1ZQVVUV5s2bJ8TGxgrz5s0T1NTUhAsXLnzx66soSuM5BQUFCYcPHxbi4uKEq1evCr6+voKampoQGRn5xa+vIinqsxIEQbh69apw9epVoXHjxkKfPn2Eq1evCjdv3hTr+ZkqeaXxnPiZKh1FfVa//faboKamJvy///f/hKSkJHF79eqV2IafqZJXGs+Jn6kvgwkkERXZ//t//08wMzMTNDQ0hEaNGgmnTp0S6wYMGCC4ubkptA8PDxcaNmwoaGhoCObm5sKqVaty9fn7778L1tbWgrq6umBjYyPs3r27tC+jwivp5zR27FihVq1agoaGhlC1alXhu+++E86dO/clLqXCK+qzApBrMzMzU2jDz1TJK+nnxM9U6SnKs3Jzc8vzWQ0YMEChT36mSl5JPyd+pr4MiSD8/6skEBERERERERWA70ASERERERGRUphAEhERERERkVKYQBIREREREZFSmEASERERERGRUphAEhERERERkVKYQBIREREREZFSmEASERERERGRUphAEhERERERkVKYQBIREREREZFSmEASERFRueXj44MuXbqUdRh5io+Ph0QiQXR0dFmHQkT0xTCBJCIiIiqizMzMsg6BiKhMMIEkIiKir4K7uztGjRqFsWPHwsDAAEZGRvj111+Rnp4OX19f6OnpoW7dujh06JB4THh4OCQSCf766y80aNAAmpqaaNq0KW7cuKHQ9+7du1GvXj1IpVKYm5tj8eLFCvXm5uaYNWsWfHx8IJPJMHjwYNSuXRsA0LBhQ0gkEri7uwMALl26hLZt26JKlSqQyWRwc3PDlStXFPqTSCRYt24dunbtCm1tbVhaWmL//v0KbW7evIkOHTpAX18fenp6aNGiBeLi4sT6kJAQ2NraQlNTEzY2Nli5cuVn32MiosIwgSQiIqKvxsaNG1GlShVcvHgRo0aNwg8//IAePXrA1dUVV65cgYeHB/r37483b94oHDdp0iQsWrQIly5dQrVq1fC///0P79+/BwBERUXBy8sLvXr1wo0bNxAUFITAwECEhoYq9LFw4ULUr18fUVFRCAwMxMWLFwEAx48fR1JSEvbs2QMAeP36NQYMGICIiAhcuHABlpaW8PT0xOvXrxX6Cw4OhpeXF65fvw5PT0/07dsXL168AAA8fPgQLVu2hKamJk6ePImoqCj4+fkhKysLALB27VpMnToVs2fPRmxsLObMmYPAwEBs3LixxO85EZECgYiIiKicGjBggNC5c2dBEATBzc1N+Oabb8S6rKwsQUdHR+jfv79YlpSUJAAQzp8/LwiCIISFhQkAhO3bt4ttnj9/LmhpaQk7duwQBEEQ+vTpI7Rt21bhvJMmTRLs7OzEfTMzM6FLly4Kbe7fvy8AEK5evVrgNWRlZQl6enrCn3/+KZYBEKZNmybup6WlCRKJRDh06JAgCIIQEBAg1K5dW8jMzMyzT1NTU+G3335TKJs5c6bg4uJSYCxERJ+LI5BERET01XBwcBD/raqqCkNDQ9jb24tlRkZGAIDk5GSF41xcXMR/V65cGdbW1oiNjQUAxMbGonnz5grtmzdvjrt37yI7O1ssc3JyUirG5ORkDBs2DFZWVpDJZJDJZEhLS0NCQkK+16KjowM9PT0x7ujoaLRo0QLq6uq5+n/69CkSExMxcOBA6OrqitusWbMUprgSEZUGtbIOgIiIiEhZnyZUEolEoUwikQAA5HJ5oX3ltBUEQfx3DkEQcrXX0dFRKkYfHx88ffoUS5cuhZmZGaRSKVxcXHItvJPXteTEraWllW//OW3Wrl2Lpk2bKtSpqqoqFSMRUXExgSQiIqIK78KFC6hVqxYA4OXLl7hz5w5sbGwAAHZ2djhz5oxC+3PnzsHKyqrAhExDQwMAFEYpASAiIgIrV66Ep6cnACAxMRHPnj0rUrwODg7YuHEj3r9/nyvRNDIygomJCe7du4e+ffsWqV8ios/FBJKIiIgqvBkzZsDQ0BBGRkaYOnUqqlSpIn6/5IQJE9CkSRPMnDkTPXv2xPnz57FixYpCVzWtVq0atLS0cPjwYdSsWROampqQyWSwsLDA5s2b4eTkhNTUVEyaNKnAEcW8jBw5EsuXL0evXr0QEBAAmUyGCxcuwNnZGdbW1ggKCsLo0aOhr6+P9u3bIyMjA5cvX8bLly8xfvz44t4mIqJC8R1IIiIiqvDmzZuHMWPGoHHjxkhKSsL+/fvFEcRGjRph586d2L59O+rXr4/p06djxowZ8PHxKbBPNTU1/PLLL1izZg1q1KiBzp07AwA2bNiAly9fomHDhujfvz9Gjx6NatWqFSleQ0NDnDx5EmlpaXBzc0Pjxo2xdu1acTRy0KBBWLduHUJDQ2Fvbw83NzeEhoaKXy1CRFRaJEJek/yJiIiIKoDw8HC0atUKL1++RKVKlco6HCKirx5HIImIiIiIiEgpTCCJiIiIiIhIKZzCSkRERERERErhCCQREREREREphQkkERERERERKYUJJBERERERESmFCSQREREREREphQkkERERERERKYUJJBERERER0f/Xfh0IAAAAAAjyt15hgLKIRSABAABYBBIAAIAlpzXdU6KXw3EAAAAASUVORK5CYII=",
      "text/plain": [
       "<Figure size 800x400 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "importances = pd.Series(best_rf.feature_importances_, index=X.columns)\n",
    "top_features = importances.sort_values(ascending=False)[:10]\n",
    "\n",
    "plt.figure(figsize=(8,4))\n",
    "sns.barplot(x=top_features, y=top_features.index)\n",
    "plt.title(\"Top 10 Important Features\")\n",
    "plt.xlabel(\"Importance\")\n",
    "plt.ylabel(\"Feature\")\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "9212bcbb-9b46-433c-ad16-15f417f89cb0",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python [conda env:base] *",
   "language": "python",
   "name": "conda-base-py"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.13.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
