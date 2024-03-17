from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from scipy.optimize import minimize
import numpy as np
import pandas as pd
import json
# Toth 함수 정의
def toth_function(x, a, b, c):
    return c * b * x / ((1 + (b * x) ** a) ** (1 / a))

# Mean squared error (MSE) calculation function
def mean_squared_error(params, x_data, y_data):
    a_val, b_val, c_val = params
    y_pred = toth_function(x_data, a_val, b_val, c_val)
    return np.mean((y_pred - y_data) ** 2)

@method_decorator(csrf_exempt)
def submit_datas(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            result_list = []

            # Process each table
            count = 0
            for table_data in data:
                count +=1
                table_name = table_data['name']
                flattened_data = [float(item.strip()) for sublist in table_data['data'] for item in sublist if item.strip()]
                x_data = np.array(flattened_data[::2], dtype=np.float64)  # Extract x values
                y_data =np.array( flattened_data[1::2] ,dtype=np.float64) # Extract y values
                print(type(x_data), y_data)
                # Check if data exists
                if len(x_data) == 0 or len(y_data) == 0:
                    return JsonResponse({'error': f"No data found for {table_name}"}, status=400)

                # Perform optimization
                result = minimize(mean_squared_error, [55, 10, 1276], args=(x_data, y_data), method='Nelder-Mead', options={'maxiter': 10000})
                optimal_a, optimal_b, optimal_c = result.x
                min_mse = result.fun * len(x_data)

                # Append results to the result_list
                result_list.append({
                    "id" : count ,
                    'table_name': table_name,
                    'mean_square_error': min_mse,
                    'parameters': {'a': optimal_a, 'b': optimal_b, 'c': optimal_c},
                    "x_data" : list(x_data),
                    "original_data" :list( y_data),
                    "fitting_data" : list(toth_function(x_data, optimal_a,optimal_b, optimal_c))
                })
            print(result_list)
            return JsonResponse(result_list, safe=False)

        except json.JSONDecodeError:
            return JsonResponse({'error': 'Invalid JSON format'}, status=400)

    else:
        return HttpResponse('Bad Request: Only POST requests are allowed', status=400)