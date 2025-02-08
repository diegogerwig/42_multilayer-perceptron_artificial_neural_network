import os
import webbrowser
import sys
import numpy as np
from colorama import Fore

def generate_html_report(results_df, metrics, X_test, model_data, skip_input=False):
    try:
        os.makedirs('./report', exist_ok=True)
        
        # Feature importance
        W1 = model_data['W'][0]
        feature_importance = np.abs(W1).mean(axis=1)
        feature_names = [
        "Radius mean", "Radius se", "Radius worst",
        "Texture mean", "Texture se", "Texture worst", 
        "Perimeter mean", "Perimeter se", "Perimeter worst",
        "Area mean", "Area se", "Area worst",
        "Smoothness mean", "Smoothness se", "Smoothness worst",
        "Compactness mean", "Compactness se", "Compactness worst",
        "Concavity mean", "Concavity se", "Concavity worst",
        "Concave points mean", "Concave points se", "Concave points worst",
        "Symmetry mean", "Symmetry se", "Symmetry worst",
        "Fractal dimension mean", "Fractal dimension se", "Fractal dimension worst"
        ]
        importance_data = list(zip(feature_names, feature_importance))
        importance_data.sort(key=lambda x: x[1], reverse=True)

        # Weights statistics
        layer_weights = []
        for i, w in enumerate(model_data['W']):
            layer_weights.append({
                'layer': i + 1,
                'shape': w.shape,
                'mean': float(np.mean(w)),
                'std': float(np.std(w)),
                'min': float(np.min(w)),
                'max': float(np.max(w))
            })

        # Generate feature importance rows
        feature_rows = []
        for name, importance in importance_data:
            width = min(100, int(importance * 100))
            feature_rows.append(f"""
                <tr class="bg-gray-800 hover:bg-gray-700">
                    <td class="px-6 py-4 whitespace-nowrap text-gray-300">{name}</td>
                    <td class="px-6 py-4">
                        <div class="w-full bg-gray-700 rounded-full h-4">
                            <div class="bg-blue-600 h-4 rounded-full" style="width: {width}%"></div>
                        </div>
                    </td>
                    <td class="px-6 py-4 text-gray-300">{importance:.4f}</td>
                </tr>""")

        # Generate weights statistics rows
        weight_rows = []
        for w in layer_weights:
            weight_rows.append(f"""
                <tr class="bg-gray-800 hover:bg-gray-700">
                    <td class="px-6 py-4 text-gray-300">Layer {w['layer']}</td>
                    <td class="px-6 py-4 text-gray-300">{w['shape']}</td>
                    <td class="px-6 py-4 text-gray-300">{w['mean']:.4f}</td>
                    <td class="px-6 py-4 text-gray-300">{w['std']:.4f}</td>
                    <td class="px-6 py-4 text-gray-300">{w['min']:.4f}</td>
                    <td class="px-6 py-4 text-gray-300">{w['max']:.4f}</td>
                </tr>""")

        # Generate table rows first
        table_rows = []
        for _, row in results_df.iterrows():
            try:
                id_val = row.name if isinstance(row.name, str) else row.iloc[0]
                actual = row.iloc[1]
                predicted = row['Predicted']
                probability = row['M_Probability']
                confidence = probability if predicted == 'M' else 1 - probability
                is_correct = actual == predicted
                
                actual_class = "bg-red-900 text-red-200" if actual == 'M' else "bg-green-900 text-green-200"
                pred_class = "bg-red-900 text-red-200" if predicted == 'M' else "bg-green-900 text-green-200"
                result_class = "text-green-400" if is_correct else "text-red-400"
                result_icon = "✓" if is_correct else "✗"
                
                row_html = f'''
                    <tr class="bg-gray-800 hover:bg-gray-700">
                        <td class="px-6 py-4 whitespace-nowrap text-gray-300">{id_val}</td>
                        <td class="px-6 py-4 whitespace-nowrap">
                            <span class="px-2 inline-flex text-xs leading-5 font-semibold rounded-full {actual_class}">{actual}</span>
                        </td>
                        <td class="px-6 py-4 whitespace-nowrap">
                            <span class="px-2 inline-flex text-xs leading-5 font-semibold rounded-full {pred_class}">{predicted}</span>
                        </td>
                        <td class="px-6 py-4 whitespace-nowrap text-gray-300">{confidence:.2%}</td>
                        <td class="px-6 py-4 whitespace-nowrap">
                            <span class="inline-flex items-center {result_class}">{result_icon}</span>
                        </td>
                    </tr>'''
                table_rows.append(row_html)
            except Exception as e:
                print(f"Error processing row: {e}", file=sys.stderr)
                continue
        
        table_rows = "\n".join(table_rows)
        
        # Extract confusion matrix values
        cm = metrics['confusion_matrix']
        tp = int(cm['true_positives'])
        tn = int(cm['true_negatives'])
        fp = int(cm['false_positives'])
        fn = int(cm['false_negatives'])
        total = tp + tn + fp + fn
        
        # Calculate metrics
        accuracy_calc = float(tp + tn) / total if total > 0 else 0.0
        precision_calc = float(tp) / (tp + fp) if (tp + fp) > 0 else 0.0
        recall_calc = float(tp) / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_calc = 2 * precision_calc * recall_calc / (precision_calc + recall_calc) if (precision_calc + recall_calc) > 0 else 0.0
        
        # Generate HTML
        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Breast Cancer Prediction Results</title>
    <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-gray-900 text-gray-100">
    <div class="container mx-auto px-4 py-8">
        <h1 class="text-3xl font-bold mb-8 text-blue-400">Breast Cancer Prediction Results</h1>
        
        <div class="bg-gray-800 p-6 rounded-lg shadow mb-8">
            <h2 class="text-xl font-bold mb-4 text-blue-400">Confusion Matrix</h2>
            <div class="relative">
                <div class="grid grid-cols-2 max-w-lg mx-auto gap-1">
                    <div class="bg-green-700 p-8 rounded">
                        <div class="text-center">
                            <div class="text-4xl font-bold text-white mb-2">{tp}</div>
                            <div class="text-white font-bold">TRUE</div>
                            <div class="text-white font-bold">POSITIVE</div>
                        </div>
                    </div>
                    <div class="bg-red-700 p-8 rounded">
                        <div class="text-center">
                            <div class="text-4xl font-bold text-white mb-2">{fn}</div>
                            <div class="text-white font-bold">FALSE</div>
                            <div class="text-white font-bold">NEGATIVE</div>
                        </div>
                    </div>
                    <div class="bg-red-700 p-8 rounded">
                        <div class="text-center">
                            <div class="text-4xl font-bold text-white mb-2">{fp}</div>
                            <div class="text-white font-bold">FALSE</div>
                            <div class="text-white font-bold">POSITIVE</div>
                        </div>
                    </div>
                    <div class="bg-green-700 p-8 rounded">
                        <div class="text-center">
                            <div class="text-4xl font-bold text-white mb-2">{tn}</div>
                            <div class="text-white font-bold">TRUE</div>
                            <div class="text-white font-bold">NEGATIVE</div>
                        </div>
                    </div>
                </div>
                <div class="mt-4 text-center text-gray-400">
                    <div>Predicted</div>
                </div>
                <div class="absolute -lef-0 top-1/2 -translate-y-1/2 transform -rotate-90 text-gray-400">
                    <div>Verified</div>
                </div>
            </div>
        </div>

        <div class="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
            <div class="bg-gray-800 p-6 rounded-lg shadow">
                <div class="text-2xl font-bold text-blue-400">{metrics['accuracy'] * 100:.2f}%</div>
                <div class="text-gray-400">Accuracy</div>
            </div>
            <div class="bg-gray-800 p-6 rounded-lg shadow">
                <div class="text-2xl font-bold text-blue-400">{metrics['precision'] * 100:.2f}%</div>
                <div class="text-gray-400">Precision</div>
            </div>
            <div class="bg-gray-800 p-6 rounded-lg shadow">
                <div class="text-2xl font-bold text-blue-400">{metrics['recall'] * 100:.2f}%</div>
                <div class="text-gray-400">Recall</div>
            </div>
            <div class="bg-gray-800 p-6 rounded-lg shadow">
                <div class="text-2xl font-bold text-blue-400">{metrics['f1'] * 100:.2f}%</div>
                <div class="text-gray-400">F1 Score</div>
            </div>
        </div>

        <div class="bg-gray-800 p-6 rounded-lg shadow mb-8">
            <h2 class="text-xl font-bold mb-4 text-blue-400">Metrics Explanation</h2>
            <div class="space-y-6">
                <div>
                    <h3 class="text-lg font-semibold text-blue-300">Accuracy: {metrics['accuracy'] * 100:.2f}%</h3>
                    <p class="text-gray-400 mb-2">Total correct predictions (true positives and true negatives) divided by total predictions.</p>
                    <div class="bg-gray-900 p-3 rounded">
                        <p class="text-sm text-gray-300">
                            (TP + TN) / Total = ({tp} + {tn}) / {total} = {accuracy_calc:.4f}
                        </p>
                    </div>
                </div>

                <div>
                    <h3 class="text-lg font-semibold text-blue-300">Precision: {metrics['precision'] * 100:.2f}%</h3>
                    <p class="text-gray-400 mb-2">True positives divided by all positive predictions. Shows how many of the predicted positives are actually positive.</p>
                    <div class="bg-gray-900 p-3 rounded">
                        <p class="text-sm text-gray-300">
                            TP / (TP + FP) = {tp} / ({tp} + {fp}) = {precision_calc:.4f}
                        </p>
                    </div>
                </div>

                <div>
                    <h3 class="text-lg font-semibold text-blue-300">Recall: {metrics['recall'] * 100:.2f}%</h3>
                    <p class="text-gray-400 mb-2">True positives divided by all actual positive cases. Shows how many of the actual positives are predicted positive.</p>
                    <div class="bg-gray-900 p-3 rounded">
                        <p class="text-sm text-gray-300">
                            TP / (TP + FN) = {tp} / ({tp} + {fn}) = {recall_calc:.4f}
                        </p>
                    </div>
                </div>

                <div>
                    <h3 class="text-lg font-semibold text-blue-300">F1 Score: {metrics['f1'] * 100:.2f}%</h3>
                    <p class="text-gray-400 mb-2">Harmonic mean of precision and recall. Represents a balance between precision and recall.</p>
                    <div class="bg-gray-900 p-3 rounded">
                        <p class="text-sm text-gray-300">
                            2 * (Precision * Recall) / (Precision + Recall) = {f1_calc:.4f}
                        </p>
                    </div>
                </div>
            </div>
        </div>

        <div class="bg-gray-800 p-6 rounded-lg shadow mb-8">
            <h2 class="text-xl font-bold mb-4 text-blue-400">Feature Importance</h2>
            <div class="overflow-x-auto">
                <table class="w-full">
                    <thead class="bg-gray-900">
                        <tr>
                            <th class="px-6 py-3 text-left text-xs font-medium text-gray-400 uppercase">Feature</th>
                            <th class="px-6 py-3 text-left text-xs font-medium text-gray-400 uppercase">Importance</th>
                            <th class="px-6 py-3 text-left text-xs font-medium text-gray-400 uppercase">Value</th>
                        </tr>
                    </thead>
                    <tbody class="divide-y divide-gray-700">
                        {''.join(feature_rows)}
                    </tbody>
                </table>
            </div>
        </div>

        <div class="bg-gray-800 p-6 rounded-lg shadow mb-8">
            <h2 class="text-xl font-bold mb-4 text-blue-400">Network Weights Statistics</h2>
            <div class="overflow-x-auto">
                <table class="w-full">
                    <thead class="bg-gray-900">
                        <tr>
                            <th class="px-6 py-3 text-left text-xs font-medium text-gray-400 uppercase">Layer</th>
                            <th class="px-6 py-3 text-left text-xs font-medium text-gray-400 uppercase">Shape</th>
                            <th class="px-6 py-3 text-left text-xs font-medium text-gray-400 uppercase">Mean</th>
                            <th class="px-6 py-3 text-left text-xs font-medium text-gray-400 uppercase">Std</th>
                            <th class="px-6 py-3 text-left text-xs font-medium text-gray-400 uppercase">Min</th>
                            <th class="px-6 py-3 text-left text-xs font-medium text-gray-400 uppercase">Max</th>
                        </tr>
                    </thead>
                    <tbody class="divide-y divide-gray-700">
                        {''.join(weight_rows)}
                    </tbody>
                </table>
            </div>
        </div>

        <div class="bg-gray-800 rounded-lg shadow overflow-hidden">
            <h2 class="text-xl font-bold p-6 text-blue-400">Detailed Predictions</h2>
            <div class="overflow-x-auto">
                <table class="w-full">
                    <thead class="bg-gray-900">
                        <tr>
                            <th class="px-6 py-3 text-left text-xs font-medium text-gray-400 uppercase">ID</th>
                            <th class="px-6 py-3 text-left text-xs font-medium text-gray-400 uppercase">Actual</th>
                            <th class="px-6 py-3 text-left text-xs font-medium text-gray-400 uppercase">Predicted</th>
                            <th class="px-6 py-3 text-left text-xs font-medium text-gray-400 uppercase">Confidence</th>
                            <th class="px-6 py-3 text-left text-xs font-medium text-gray-400 uppercase">Result</th>
                        </tr>
                    </thead>
                    <tbody class="divide-y divide-gray-700">
                        {table_rows}
                    </tbody>
                </table>
            </div>
        </div>
    </div>
</body>
</html>"""

        # Save HTML report
        report_path = os.path.join('./report', 'prediction_report.html')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        # Only try to open in browser if not in skip_input mode
        absolute_path = os.path.abspath(report_path)
        if not skip_input:
            try:
                webbrowser.open('file://' + absolute_path)
            except Exception as e:
                print(f"Warning: Could not open browser: {e}", file=sys.stderr)

        return report_path

    except Exception as e:
        print(f"Error generating report: {e}", file=sys.stderr)
        return None