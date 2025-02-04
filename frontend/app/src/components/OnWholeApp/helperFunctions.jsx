const testMetricsOptions = [
  { value: "mse", label: "Mean Squared Error" },
  { value: "mae", label: "Mean Absolute Error" },
  { value: "rmse", label: "Root Mean Squared Error" },
  { value: "msle", label: "Mean Squared Logarithmic Error" },
  { value: "mape", label: "Mean Absolute Percentage Error" },
  { value: "accuracy", label: "Accuracy" },
  { value: "precision", label: "Precision" },
  { value: "recall", label: "Recall" },
  { value: "f1_score", label: "F1 Score" },
  { value: "auc", label: "Area Under Curve (AUC)" },
  { value: "log_loss", label: "Log Loss" },
  { value: "r2_score", label: "R^2 Score" },
];

// Multiselect Component for Test Metrics
const TestMetricsMultiselect = ({ register }) => {
  return (
    <div className="mt-2">
      <h5>Select Test Metrics:</h5>
      {testMetricsOptions.map((option) => (
        <div key={option.value} className="form-check form-check-inline">
          <input
            className="form-check-input"
            type="checkbox"
            value={option.value}
            id={`test-metric-${option.value}`}
            {...register("model_info.test_metrics")}
          />
          <label
            className="form-check-label"
            htmlFor={`test-metric-${option.value}`}
          >
            {option.label}
          </label>
        </div>
      ))}
    </div>
  );
};

export default TestMetricsMultiselect;
// multi import and export
// export { TestMetricsMultiselect, LossSelect, ActivationSelect };
// import { TestMetricsMultiselect, LossSelect, ActivationSelect } from './Components';
